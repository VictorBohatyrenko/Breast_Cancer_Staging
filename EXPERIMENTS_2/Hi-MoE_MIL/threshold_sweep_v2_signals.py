#!/usr/bin/env python
"""
Версія 2 threshold-свіпу для RoutedDTFDASMIL. Закриває 4 з 5 пунктів
"що реально варто зробити далі" з project-документа (усе, крім прогону на C17
-- те просто треба запустити цим самим скриптом з C17-конфігами):

1. Бейзлайн простої confidence замість tier1-ансамблю (`--signal cheap_conf`)
   -- перевіряє, чи tier1-disagreement взагалі кращий за тривіальний сигнал.
2. Неперервні сигнали невпевненості (`entropy`, `mi` / epistemic MI,
   `std_winner`) поруч зі старим дискретним `disagree` -- без перетренування.
3. Ескалація за ПЕРЦЕНТИЛЕМ (top-K% найневпевненіших), а не за абсолютним
   порогом -- знімає проблему квантування M=8 (при disagree це й досі
   квантовано ВСЕРЕДИНІ сіда, але тепер можна просити будь-який K%, і
   усереднення по 10 сідах згладжує решту).
4. У виводі тепер ЯВНО друкуються acc/f1 cheap-only і expensive-only
   (раніше були лише p-values) -- потрібно для таблиці в статті.
5. Bootstrap (resample по (seed, slide) парах) для довірчих інтервалів --
   компенсує малу статистичну потужність при n=10 сідів.
6. [ДОДАНО 2026-09-28] `random`-бейзлайн -- ескалація ВИПАДКОВИХ pct%
   слайдів (без жодного сигналу) + прямий парний t-test "цільовий сигнал
   vs random" на тому самому %esc. Це відповідає на питання "може, весь
   виграш просто тому, що дорога модель у середньому краща, а не тому, що
   ми правильно ВИБИРАЄМО, які слайди ескалювати?" -- якщо сигнал не
   значуще кращий за random, контрибуції немає.

Сигнали невпевненості (усе рахується з tier1_slide_preds [M, n_class],
які вже й так повертає forward()):
    disagree    -- стара міра: частка bags, чий argmax != мода (квантована).
    entropy     -- ентропія УСЕРЕДНЕНОГО по M bags розподілу (total
                   predictive entropy ансамблю). Неперервна.
    mi          -- epistemic uncertainty (BALD-style mutual information):
                   H(mean_prob) - mean(H(prob_i)). Це принциповий
                   неперервний аналог "disagreement" -- висока MI означає
                   bags розходяться, а не просто кожен окремо невпевнений.
    std_winner  -- std ймовірності класу-переможця (argmax mean_prob) серед
                   M bags.
    cheap_conf  -- 1 - max_softmax(фінального cheap_pred). НЕ використовує
                   tier1-ансамбль взагалі -- це "тривіальний" бейзлайн-роутер,
                   з яким і порівнюємо tier1-based сигнали.

Запуск (приклад для BRACS -- ті самі --config_cheap/--config_expensive/
--pattern_cheap/--pattern_expensive/--ckpt_name, що й у v1):
    python threshold_sweep_v2_signals.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --pattern_cheap "ckpt/bracs_dtfd_v2_seed*" \
        --pattern_expensive "ckpt/bracs_dtfd_v2_uni2h_seed*" \
        --ckpt_name checkpoint-best.pth \
        --escalate_pct 5,10,15,20,25,30,40,50 \
        --bootstrap 2000

Для C17 -- ті самі прапори, підстав C17-конфіги/паттерни (див. головний
project-документ, розділ 4, чи run_dtfd_v2_uni2h_both.sh).
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
import yaml
from scipy import stats as sstats
from sklearn.metrics import accuracy_score, f1_score

from utils.utils import Struct
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIGNALS = ["disagree", "entropy", "mi", "std_winner", "cheap_conf"]
# "random" рахується ОКРЕМО (не з forward pass, а як контрольний бейзлайн:
# випадкові pct% слайдів, без жодного сигналу). Це найважливіша перевірка --
# якщо цільовий сигнал (mi/std_winner) НЕ кращий за random при тому самому
# %esc, то весь виграш routed-моделі пояснюється просто тим, що "дорога
# модель у середньому краща", а не тим, ЯКІ саме слайди обрані для ескалації.
ALL_SIGNAL_NAMES = SIGNALS + ["random"]


class PairedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, split_csv, feature_dir_cheap, feature_dir_expensive):
        import pandas as pd
        self.df = pd.read_csv(split_csv)
        self.feature_dir_cheap = feature_dir_cheap
        self.feature_dir_expensive = feature_dir_expensive

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = int(row["label"])
        data_c = torch.load(os.path.join(self.feature_dir_cheap, f"{slide_id}.pt"),
                             map_location="cpu", weights_only=False)
        data_e = torch.load(os.path.join(self.feature_dir_expensive, f"{slide_id}.pt"),
                             map_location="cpu", weights_only=False)
        return {'feats_cheap': data_c['features'], 'coords_cheap': data_c['coords'].float(),
                'feats_expensive': data_e['features'], 'coords_expensive': data_e['coords'].float(),
                'label': label, 'slide_id': slide_id}


def load_conf(config_path, pretrain, n_token=8):
    with open(config_path, "r") as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    c['pretrain'] = pretrain
    c['n_token'] = n_token
    conf = Struct(**c)
    if pretrain == 'medical_ssl':
        conf.D_feat, conf.D_inner = 384, 128
    elif pretrain == 'UNI2h':
        conf.D_feat, conf.D_inner = 1536, 768
    return conf


def load_model(conf, ckpt_path, M):
    model = DTFD_ASMIL_v2(conf, n_token=conf.n_token, M=M).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model'] if 'model' in state else state
    model.load_state_dict(sd)
    model.eval()
    return model


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def all_signals_of(tier1_slide_preds, cheap_pred):
    """tier1_slide_preds: [M, n_class] логіти. cheap_pred: [1, n_class] логіти
    фінального (tier2) вердикту дешевої моделі. Повертає dict {signal_name: float}."""
    probs = torch.softmax(tier1_slide_preds, dim=-1).cpu().numpy()  # [M, n_class]
    argmax = probs.argmax(axis=-1)
    mode = np.bincount(argmax).argmax()
    disagree = float((argmax != mode).mean())

    mean_prob = probs.mean(axis=0)                      # [n_class]
    h_of_mean = float(entropy(mean_prob))                # total predictive entropy
    mean_of_h = float(entropy(probs).mean())              # середня "особиста" ентропія bag'ів
    mi = h_of_mean - mean_of_h                            # epistemic (BALD-style)

    winner = mean_prob.argmax()
    std_winner = float(probs[:, winner].std())

    cheap_probs = torch.softmax(cheap_pred, dim=-1).cpu().numpy().reshape(-1)
    cheap_conf_uncertainty = float(1.0 - cheap_probs.max())

    return {
        "disagree": disagree,
        "entropy": h_of_mean,
        "mi": mi,
        "std_winner": std_winner,
        "cheap_conf": cheap_conf_uncertainty,
    }


def compute_seed_data(conf_cheap, conf_exp, cheap_ckpt, expensive_ckpt, M, dataset, n_repeats=5):
    """n_repeats > 1: _split_pseudo_bags() у моделі використовує
    torch.randperm БЕЗ фіксованого сіда -- кожен forward pass наново
    випадково розбиває тайли на псевдо-bags, тому одиночний прогін дає
    "шумну" оцінку і cheap_pred/сигналів, і навіть саме acc/f1 (переконались
    емпірично: два прогони того самого коду й чекпоінтів дали різні
    cheap-only/expensive-only числа). Тут це виправляємо MC-усередненням:
    кожен слайд проганяється n_repeats разів (з різним, але відтворюваним
    для цього запуску сідом на кожен repeat), softmax-ймовірності і сигнали
    усереднюються по repeats -- це прибирає шум від випадкового bagging, не
    чіпаючи саму архітектуру моделі."""
    cheap_model = load_model(conf_cheap, cheap_ckpt, M)
    expensive_model = load_model(conf_exp, expensive_ckpt, M)

    n = len(dataset)
    signals_sum = {name: np.zeros(n) for name in SIGNALS}
    cheap_prob_sum, exp_prob_sum, trues = None, None, [None] * n

    with torch.no_grad():
        for r in range(n_repeats):
            torch.manual_seed(1000 * r + 7)  # відтворюваність between запусками скрипта
            for i, item in enumerate(dataset):
                feats_c = item['feats_cheap'].unsqueeze(0).to(device)
                feats_e = item['feats_expensive'].unsqueeze(0).to(device)
                coords_c = item['coords_cheap'].unsqueeze(0).to(device)
                coords_e = item['coords_expensive'].unsqueeze(0).to(device)

                cp, t1preds, _ = cheap_model(feats_c, coords=coords_c)
                ep, _, _ = expensive_model(feats_e, coords=coords_e)

                if cheap_prob_sum is None:
                    n_class = cp.shape[-1]
                    cheap_prob_sum = np.zeros((n, n_class))
                    exp_prob_sum = np.zeros((n, n_class))

                sig = all_signals_of(t1preds, cp)
                for name in SIGNALS:
                    signals_sum[name][i] += sig[name]
                cheap_prob_sum[i] += torch.softmax(cp, dim=-1).cpu().numpy().reshape(-1)
                exp_prob_sum[i] += torch.softmax(ep, dim=-1).cpu().numpy().reshape(-1)
                if r == 0:
                    trues[i] = item['label']

    del cheap_model, expensive_model
    torch.cuda.empty_cache()

    out = {name: (vals / n_repeats) for name, vals in signals_sum.items()}
    out["cheap_pred"] = cheap_prob_sum.argmax(axis=-1)
    out["exp_pred"] = exp_prob_sum.argmax(axis=-1)
    out["true"] = np.array(trues)
    return out


def add_random_signal(per_seed_data, base_seed=12345):
    """Додає d['random'] -- незалежний, відтворюваний випадковий скор для
    кожного сіда (RNG seeded по (base_seed, seed), а не по torch/model seed,
    щоб не збігатись випадково з чимось значущим)."""
    for seed, d in per_seed_data.items():
        rng = np.random.default_rng(base_seed + seed)
        d["random"] = rng.random(len(d["true"]))


def escalate_top_pct(score, pct):
    """Ескалює top ceil(pct% * N) слайдів за спаданням score (а не за
    абсолютним порогом) -- дає РІВНО потрібний % ескалацій незалежно від
    того, наскільки дискретний/квантований сигнал score."""
    n = len(score)
    k = int(np.ceil(pct / 100.0 * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    order = np.argsort(-score, kind="stable")
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    return flagged


def bootstrap_ci(pooled_correct_routed, pooled_correct_baseline, n_boot=2000, seed=0):
    """Bootstrap 95% ДІ для Δaccuracy = mean(routed) - mean(baseline), resample
    по (seed, slide) парах (spарені -- той самий підмножина індексів для обох)."""
    rng = np.random.default_rng(seed)
    n = len(pooled_correct_routed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = pooled_correct_routed[idx].mean() - pooled_correct_baseline[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--pattern_cheap", required=True)
    ap.add_argument("--pattern_expensive", required=True)
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--escalate_pct", default="5,10,15,20,25,30,40,50")
    ap.add_argument("--signals", default=",".join(ALL_SIGNAL_NAMES),
                     help="комою: підмножина з " + ",".join(ALL_SIGNAL_NAMES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--n_repeats", type=int, default=5,
                     help="MC-усереднення forward passes на слайд, щоб прибрати "
                          "шум від нефіксованого random bagging у моделі "
                          "(1 = стара поведінка, без усереднення)")
    ap.add_argument("--bootstrap", type=int, default=0,
                     help="0 = вимкнено; інакше -- к-сть bootstrap resamples "
                          "для ДІ на сигналі/pct з найкращим f1 (повільно, "
                          "робиться один раз наприкінці)")
    args = ap.parse_args()

    pcts = [float(x) for x in args.escalate_pct.split(",")]
    signals_to_run = [s.strip() for s in args.signals.split(",")]
    conf_cheap = load_conf(args.config_cheap, "medical_ssl", args.n_token)
    conf_exp = load_conf(args.config_expensive, "UNI2h", args.n_token)
    M = args.n_token

    split_csv = os.path.join(conf_cheap.splits_dir, f'{args.split}.csv')
    feature_dir_cheap = conf_cheap.train_dir if args.split != 'test' else conf_cheap.test_dir
    feature_dir_exp = conf_exp.train_dir if args.split != 'test' else conf_exp.test_dir
    dataset = PairedFeatureDataset(split_csv, feature_dir_cheap, feature_dir_exp)

    def extract_seed(dirname):
        m = re.search(r'seed(\d+)$', dirname)
        return int(m.group(1)) if m else None

    cheap_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_cheap))}
    exp_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_expensive))}
    common_seeds = sorted(set(cheap_dirs) & set(exp_dirs))
    print(f"Спільні сіди: {common_seeds}")

    per_seed_data = {}
    for seed in common_seeds:
        cheap_ckpt = os.path.join(cheap_dirs[seed], args.ckpt_name)
        exp_ckpt = os.path.join(exp_dirs[seed], args.ckpt_name)
        if not (os.path.exists(cheap_ckpt) and os.path.exists(exp_ckpt)):
            print(f"[skip] seed{seed}")
            continue
        print(f"Прогін seed{seed} ({args.n_repeats}x MC-усереднення) ...")
        per_seed_data[seed] = compute_seed_data(conf_cheap, conf_exp, cheap_ckpt, exp_ckpt, M,
                                                 dataset, n_repeats=args.n_repeats)

    add_random_signal(per_seed_data)

    # --- Baseline-таблиця: cheap-only і expensive-only (не залежить від порогу) ---
    acc_c, f1_c, acc_e, f1_e = [], [], [], []
    for seed, d in per_seed_data.items():
        acc_c.append(accuracy_score(d["true"], d["cheap_pred"]))
        f1_c.append(f1_score(d["true"], d["cheap_pred"], average='macro'))
        acc_e.append(accuracy_score(d["true"], d["exp_pred"]))
        f1_e.append(f1_score(d["true"], d["exp_pred"], average='macro'))
    print("\n=== Бейзлайни (не залежать від порогу/сигналу) ===")
    print(f"cheap-only:     acc={np.mean(acc_c):.4f}±{np.std(acc_c, ddof=1):.3f}  "
          f"f1={np.mean(f1_c):.4f}±{np.std(f1_c, ddof=1):.3f}")
    print(f"expensive-only: acc={np.mean(acc_e):.4f}±{np.std(acc_e, ddof=1):.3f}  "
          f"f1={np.mean(f1_e):.4f}±{np.std(f1_e, ddof=1):.3f}")

    best_by_f1 = None  # (f1_mean, signal, pct, pooled_correct_routed, pooled_correct_cheap)
    acc_by_signal_pct = {}  # (signal, pct) -> acc_arr [n_seeds], для прямого порівняння сигналів

    for signal_name in signals_to_run:
        print(f"\n=== Сигнал: {signal_name} ===")
        print(f"{'pct_esc':>8s}  {'acc_routed':>11s}  {'f1_routed':>10s}  "
              f"{'p(vs cheap)':>12s}  {'p(vs exp)':>10s}")
        for pct in pcts:
            acc_list, f1_list, real_esc_list = [], [], []
            acc_cheap_list, acc_exp_list = [], []
            pooled_routed, pooled_cheap = [], []
            for seed, d in per_seed_data.items():
                score = d[signal_name]
                flagged = escalate_top_pct(score, pct)
                routed_pred = np.where(flagged, d["exp_pred"], d["cheap_pred"])
                acc_list.append(accuracy_score(d["true"], routed_pred))
                f1_list.append(f1_score(d["true"], routed_pred, average='macro'))
                real_esc_list.append(100 * flagged.mean())
                acc_cheap_list.append(accuracy_score(d["true"], d["cheap_pred"]))
                acc_exp_list.append(accuracy_score(d["true"], d["exp_pred"]))
                pooled_routed.extend((routed_pred == d["true"]).astype(int).tolist())
                pooled_cheap.extend((d["cheap_pred"] == d["true"]).astype(int).tolist())

            acc_arr, f1_arr = np.array(acc_list), np.array(f1_list)
            acc_c_arr, acc_e_arr = np.array(acc_cheap_list), np.array(acc_exp_list)
            _, p_vs_cheap = sstats.ttest_rel(acc_arr, acc_c_arr)
            _, p_vs_exp = sstats.ttest_rel(acc_arr, acc_e_arr)

            print(f"{pct:8.1f}  {acc_arr.mean():.4f}±{acc_arr.std(ddof=1):.3f}  "
                  f"{f1_arr.mean():.4f}±{f1_arr.std(ddof=1):.3f}  "
                  f"{p_vs_cheap:12.4g}  {p_vs_exp:10.4g}   "
                  f"(факт. esc={np.mean(real_esc_list):.1f}%)")

            acc_by_signal_pct[(signal_name, pct)] = acc_arr

            if best_by_f1 is None or f1_arr.mean() > best_by_f1[0]:
                best_by_f1 = (f1_arr.mean(), signal_name, pct,
                              np.array(pooled_routed), np.array(pooled_cheap))

    # --- НАЙВАЖЛИВІШИЙ блок: чи цільові сигнали справді кращі за random  ---
    # ескалацію ТОГО Ж %, а не просто за те, що "дорога модель у середньому
    # краща"? Парний t-test на тих самих 10 сідах (той самий acc_arr, що й
    # вище -- тому парність коректна: для кожного сіда порівнюємо
    # acc_routed(сигнал) vs acc_routed(random) при однаковому pct.
    if "random" in signals_to_run:
        print("\n=== Прямі порівняння: цільовий сигнал vs random-ескалація "
              "(той самий %esc) ===")
        print(f"{'signal':>12s}  {'pct':>6s}  {'acc(signal)':>12s}  "
              f"{'acc(random)':>12s}  {'Δacc':>8s}  {'p(paired-t)':>12s}")
        for signal_name in signals_to_run:
            if signal_name == "random":
                continue
            for pct in pcts:
                key_s = (signal_name, pct)
                key_r = ("random", pct)
                if key_s not in acc_by_signal_pct or key_r not in acc_by_signal_pct:
                    continue
                a_s = acc_by_signal_pct[key_s]
                a_r = acc_by_signal_pct[key_r]
                _, p = sstats.ttest_rel(a_s, a_r)
                delta = a_s.mean() - a_r.mean()
                print(f"{signal_name:>12s}  {pct:6.1f}  {a_s.mean():12.4f}  "
                      f"{a_r.mean():12.4f}  {delta:+8.4f}  {p:12.4g}")
        print("(p<0.05 і Δacc>0 тут -- це і є доказ, що ВИБІР слайдів на основі")
        print(" сигналу дає щось понад просту заміну випадкових pct% на дорогу модель)")

    if args.bootstrap > 0 and best_by_f1 is not None:
        f1_val, sig_name, pct, pooled_routed, pooled_cheap = best_by_f1
        print(f"\n=== Bootstrap ДІ (n={args.bootstrap}) для найкращого f1: "
              f"signal={sig_name}, pct_esc={pct} ===")
        mean_d, lo, hi = bootstrap_ci(pooled_routed, pooled_cheap, n_boot=args.bootstrap)
        print(f"Δaccuracy (routed - cheap-only) = {mean_d:+.4f}  95% ДІ=[{lo:+.4f}, {hi:+.4f}]")
        if lo > 0:
            print("  -> ДІ повністю > 0: значущий виграш над cheap-only при цьому pct/сигналі.")
        elif hi < 0:
            print("  -> ДІ повністю < 0: значуще ГІРШЕ за cheap-only.")
        else:
            print("  -> ДІ перетинає 0: ще не значуще на наявній вибірці.")


if __name__ == "__main__":
    main()
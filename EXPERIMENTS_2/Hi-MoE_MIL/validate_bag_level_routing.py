#!/usr/bin/env python
"""
BAG-РІВНЕВА перевірка гіпотези для нової ідеї (Tier1.5 з контекстом).

Наша поточна routed-архітектура ескалює ЦІЛИЙ слайд, якщо він у середньому
непевний. Нова ідея: ескалювати лише КОНКРЕТНІ bags усередині слайду, які
самі по собі найневпевненіші -- і обробляти їх окремим модулем Tier1.5,
що додатково бачить дистильований контекст решти bags.

Перш ніж писати й тренувати Tier1.5 (важкий, дорогий крок -- новий
навчальний модуль, а не просто композиція готових моделей), варто дешево
перевірити ПЕРЕДУМОВУ: чи взагалі bags з високою "особистою" невпевненістю
виграють більше від дорогих (UNI2h) фіч, ніж впевнені bags? Якщо так --
Tier1.5 має сенс будувати. Якщо ні -- ідея не спрацює на bag-рівні так само,
як не спрацював би слайд-рівень без сигналу (ми це вже перевіряли).

Метод:
    1. Для кожного слайду детерміновано (тим самим сідом для cheap і
       expensive) ділимо тайли на M=8 bags -- ЗОВНІШНЬО, а не всередині
       моделі (`_split_pseudo_bags`), щоб точно знати, які тайли в якому
       bag'у для ОБОХ backbone одночасно.
    2. Кожен bag проганяємо через `cheap_model.tier1` (з cheap-фічами) і
       ОКРЕМО через `expensive_model.tier1` (з expensive-фічами ТИХ САМИХ
       тайлів) -- напряму, минаючи повний forward() і Tier2.
    3. "Особиста" невпевненість bag'а = ентропія softmax його власного
       cheap-передбачення (bag ще не бачив інших bags -- це справді
       lokal, "власна" непевність, а не розбіжність між bags, як раніше).
    4. Proxy ground truth для bag'а = мітка слайду (те саме припущення, що
       й tier1_weight-дистиляція в тренуванні -- bag "правильний", якщо
       його argmax == мітка слайду).
    5. Порівнюємо: (а) acc(expensive bag) - acc(cheap bag) окремо на
       top-K% найневпевніших bags і на решті (McNemar, як і раніше);
       (б) чи ВИБІР bags за невпевненістю кращий за випадковий вибір
       того самого числа bags (той самий random-контроль, що врятував нас
       на слайд-рівні).

Запуск (з кореня Hi-MoE_MIL, той самий venv):
    python validate_bag_level_routing.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --pattern_cheap "ckpt/bracs_dtfd_v2_seed*" \
        --pattern_expensive "ckpt/bracs_dtfd_v2_uni2h_seed*" \
        --ckpt_name checkpoint-best.pth \
        --escalate_pct_bags 10,20,30,40,50 \
        --n_repeats 3
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
import yaml
from scipy import stats as sstats

from utils.utils import Struct
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def split_bags_externally(n_tiles, M, seed):
    """Той самий принцип, що й `_split_pseudo_bags` у моделі (випадковий
    рівномірний поділ), але зовнішньо і З ФІКСОВАНИМ сідом -- щоб ОДНА Й ТА
    Ж партиція тайлів застосовувалась і до cheap-, і до expensive-фіч того
    самого слайду (інакше "bag 3 в cheap" і "bag 3 в expensive" були б
    різними фізичними тайлами, і порівняння втратило б сенс)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_tiles, generator=g)
    m_eff = min(M, n_tiles)
    return torch.chunk(perm, m_eff)


def compute_bag_level_data(conf_cheap, conf_exp, cheap_ckpt, expensive_ckpt, M, dataset, n_repeats=3):
    """Для КОЖНОГО (слайд, bag, repeat) рахує: own_uncertainty (з cheap
    bag'а), correct_cheap, correct_exp (proxy: argmax == slide label).
    Повертає плоскі масиви (по всіх слайдах і repeats разом) -- це вже
    "рядки" для аналізу, аналогічно pooled_flag/pooled_unflag у
    uni2h_fixes_disagreement.py."""
    cheap_model = load_model(conf_cheap, cheap_ckpt, M)
    expensive_model = load_model(conf_exp, expensive_ckpt, M)

    rows_uncertainty, rows_correct_cheap, rows_correct_exp = [], [], []
    rows_slide_idx, rows_bag_idx = [], []

    with torch.no_grad():
        for r in range(n_repeats):
            for slide_i, item in enumerate(dataset):
                feats_c = item['feats_cheap'].to(device)       # [N, D_feat_cheap]
                feats_e = item['feats_expensive'].to(device)   # [N, D_feat_exp]
                coords_c = item['coords_cheap'].to(device)
                coords_e = item['coords_expensive'].to(device)
                label = item['label']
                n_tiles = feats_c.shape[0]

                if feats_c.shape[0] != feats_e.shape[0]:
                    # cheap і expensive фічі мають різну к-сть тайлів -- партиції
                    # неспівставні для цього слайду, пропускаємо.
                    continue

                chunks = split_bags_externally(n_tiles, M, seed=1000 * r + slide_i)

                for bag_i, idx in enumerate(chunks):
                    if len(idx) == 0:
                        continue
                    bag_feats_c = feats_c[idx].unsqueeze(0)     # [1, n_i, D_feat_cheap]
                    bag_feats_e = feats_e[idx].unsqueeze(0)
                    bag_coords_c = coords_c[idx].unsqueeze(0)
                    bag_coords_e = coords_e[idx].unsqueeze(0)

                    # Прямий виклик tier1-підмодуля, МИНАЮЧИ Tier2 -- нас
                    # цікавить лише "думка" цього одного bag'а.
                    _, cheap_bag_pred, _, _ = cheap_model.tier1(
                        bag_feats_c, coords=bag_coords_c, return_feats=True)
                    _, exp_bag_pred, _, _ = expensive_model.tier1(
                        bag_feats_e, coords=bag_coords_e, return_feats=True)

                    probs_c = torch.softmax(cheap_bag_pred, dim=-1).cpu().numpy().reshape(-1)
                    own_uncertainty = float(entropy(probs_c))

                    correct_c = int(probs_c.argmax() == label)
                    probs_e = torch.softmax(exp_bag_pred, dim=-1).cpu().numpy().reshape(-1)
                    correct_e = int(probs_e.argmax() == label)

                    rows_uncertainty.append(own_uncertainty)
                    rows_correct_cheap.append(correct_c)
                    rows_correct_exp.append(correct_e)
                    rows_slide_idx.append(slide_i)
                    rows_bag_idx.append(bag_i)

    del cheap_model, expensive_model
    torch.cuda.empty_cache()

    return {
        "uncertainty": np.array(rows_uncertainty),
        "correct_cheap": np.array(rows_correct_cheap),
        "correct_exp": np.array(rows_correct_exp),
        "slide_idx": np.array(rows_slide_idx),
    }


def mcnemar(correct_a, correct_b, label):
    correct_a, correct_b = np.array(correct_a), np.array(correct_b)
    n = len(correct_a)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        print(f"{label}: n={n} acc_a={correct_a.mean():.3f} acc_b={correct_b.mean():.3f} -- b=c=0")
        return
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - sstats.chi2.cdf(stat, df=1)
    print(f"{label}: n={n:5d}  acc_cheap={correct_a.mean():.4f}  acc_exp={correct_b.mean():.4f}  "
          f"(cheap-only-right={b}, exp-only-right={c})  McNemar p={p:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--pattern_cheap", required=True)
    ap.add_argument("--pattern_expensive", required=True)
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--escalate_pct_bags", default="10,20,30,40,50",
                     help="top-K%% НАЙНЕВПЕВНЕНІШИХ bags (пул усіх bags усіх "
                          "слайдів разом), для яких перевіряємо виграш від "
                          "expensive і порівнюємо з random-вибором bags")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--n_repeats", type=int, default=3)
    args = ap.parse_args()

    pcts = [float(x) for x in args.escalate_pct_bags.split(",")]
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

    all_data = []
    for seed in common_seeds:
        cheap_ckpt = os.path.join(cheap_dirs[seed], args.ckpt_name)
        exp_ckpt = os.path.join(exp_dirs[seed], args.ckpt_name)
        if not (os.path.exists(cheap_ckpt) and os.path.exists(exp_ckpt)):
            print(f"[skip] seed{seed}")
            continue
        print(f"Прогін seed{seed} (bag-рівень, {args.n_repeats}x) ...")
        d = compute_bag_level_data(conf_cheap, conf_exp, cheap_ckpt, exp_ckpt, M, dataset,
                                    n_repeats=args.n_repeats)
        all_data.append(d)

    uncertainty = np.concatenate([d["uncertainty"] for d in all_data])
    correct_cheap = np.concatenate([d["correct_cheap"] for d in all_data])
    correct_exp = np.concatenate([d["correct_exp"] for d in all_data])
    n_bags_total = len(uncertainty)
    print(f"\nВсього (сід x слайд x bag x repeat) рядків: {n_bags_total}")
    print(f"acc_cheap (усі bags) = {correct_cheap.mean():.4f}   "
          f"acc_exp (усі bags) = {correct_exp.mean():.4f}")

    print("\n=== McNemar: flagged (high own-uncertainty) vs unflagged bags ===")
    for pct in pcts:
        k = int(np.ceil(pct / 100.0 * n_bags_total))
        order = np.argsort(-uncertainty, kind="stable")
        flagged = np.zeros(n_bags_total, dtype=bool)
        flagged[order[:k]] = True
        print(f"\n-- top {pct}% найневпевненіших bags (n_flagged={flagged.sum()}) --")
        mcnemar(correct_cheap[flagged], correct_exp[flagged], "FLAGGED  ")
        mcnemar(correct_cheap[~flagged], correct_exp[~flagged], "UNFLAGGED")

    print("\n=== Головний тест: чи ВИБІР bags за невпевненістю кращий за random ===")
    print("(ВИПРАВЛЕНО: порівнюємо ПРИРІСТ exp-cheap на flagged bags проти")
    print(" такого самого приросту на random bags -- НЕ абсолютний рівень")
    print(" accuracy, бо flagged bags апріорі важчі й матимуть нижчу абс.")
    print(" accuracy навіть з дорогою моделлю -- це очікувано і нічого не")
    print(" каже про користь ескалації самої по собі.)")
    rng = np.random.default_rng(777)
    print(f"{'pct':>6s}  {'Δacc(flagged)':>14s}  {'Δacc(random)':>14s}  "
          f"{'Δ(flagged-random)':>18s}  {'p(перест.тест)':>14s}")
    for pct in pcts:
        k = int(np.ceil(pct / 100.0 * n_bags_total))
        order = np.argsort(-uncertainty, kind="stable")
        flagged_idx = order[:k]
        delta_flagged = correct_exp[flagged_idx].mean() - correct_cheap[flagged_idx].mean()

        # permutation test: 2000 випадкових виборів k bags, дивимось на
        # РОЗПОДІЛ delta_random = acc_exp(random) - acc_cheap(random) і де в
        # ньому лежить наш спостережуваний delta_flagged (one-sided: чи
        # рідко трапляється delta_random >= delta_flagged?)
        n_perm = 2000
        random_deltas = np.empty(n_perm)
        for p_i in range(n_perm):
            rand_idx = rng.choice(n_bags_total, size=k, replace=False)
            random_deltas[p_i] = correct_exp[rand_idx].mean() - correct_cheap[rand_idx].mean()
        p_value = float((random_deltas >= delta_flagged).mean())
        print(f"{pct:6.1f}  {delta_flagged:14.4f}  {random_deltas.mean():14.4f}  "
              f"{delta_flagged - random_deltas.mean():+18.4f}  {p_value:14.4g}")

    print("\n(Якщо для розумних pct% p(перест.тест) << 0.05 і Δ(flagged-random)>0 -- це і є")
    print(" сигнал, що варто будувати Tier1.5: конкретні НЕВПЕВНЕНІ bags дійсно")
    print(" виграють від expensive-фіч більше, ніж випадково обрані bags.")
    print(" Якщо ні -- bag-рівнева версія ідеї поки не підтверджена, і")
    print(" інвестувати час у новий тренований модуль Tier1.5 зарано.)")


if __name__ == "__main__":
    main()
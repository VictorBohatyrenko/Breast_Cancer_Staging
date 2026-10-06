#!/usr/bin/env python
"""
Перевіряє ключову гіпотезу для "routing"-архітектури: чи UNI2h-backbone
покращує результат САМЕ на тих слайдах, де ViT-S/DINO-версія DTFD-ASMIL v2
мала високий tier1_disagreement (тобто де ViT-S/DINO сама "не впевнена"),
а не рівномірно по всьому датасету.

Для кожного сіда (той самий --seed у ViT-S/DINO і UNI2h чекпоінтах):
  1. Прогін ViT-S/DINO чекпоінта -> pred_vits, tier1_disagreement, true
  2. Прогін UNI2h чекпоінта (та сама архітектура, інший backbone) -> pred_uni2h
  3. Розбиття тестових слайдів на flagged (disagreement >= поріг) / unflagged
  4. Порівняння accuracy(UNI2h) - accuracy(ViT-S/DINO) окремо на flagged і
     unflagged -- якщо ефект концентрований, Δacc на flagged суттєво
     більший, ніж на unflagged (перевіряється paired t-test по 10 сідах).

Запуск (з кореня Hi-MoE_MIL, з активованим venv):
    python uni2h_fixes_disagreement.py \
        --config_vits config/bracs_vits_dino_config_100ep.yml \
        --config_uni2h config/bracs_uni2h_config.yml \
        --pattern_vits "ckpt/bracs_dtfd_v2_seed*" \
        --pattern_uni2h "ckpt/bracs_dtfd_v2_uni2h_seed*" \
        --ckpt_name checkpoint-best.pth \
        --threshold 0.1
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
from pt_dataset import PTClassificationDataset
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def run_inference(model, loader, want_disagreement):
    """Повертає dict idx (порядковий номер рядка в test.csv, НЕ slide_id --
    pt_dataset.py.__getitem__ повертає (feats, label, coords), без slide_id
    взагалі) -> {pred, true[, disagreement]}. Коректно, бо обидва лоадери
    (ViT-S/DINO і UNI2h) читають ТОЙ САМИЙ test.csv з shuffle=False, тому
    порядок рядків гарантовано однаковий між ними."""
    out = {}
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image_patches, label = batch[0].to(device), batch[1]
            final_pred, tier1_slide_preds, _ = model(image_patches)
            pred = int(final_pred.argmax(dim=-1).item())
            true = int(label.item() if torch.is_tensor(label) else label)
            entry = {'pred': pred, 'true': true}
            if want_disagreement:
                t1_probs = torch.softmax(tier1_slide_preds, dim=-1)
                t1_argmax = t1_probs.argmax(dim=-1)
                mode_class = torch.mode(t1_argmax).values
                entry['disagreement'] = (t1_argmax != mode_class).float().mean().item()
            out[idx] = entry
    return out


def extract_seed(dirname):
    m = re.search(r'seed(\d+)$', dirname)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_vits", required=True)
    ap.add_argument("--config_uni2h", required=True)
    ap.add_argument("--pattern_vits", required=True)
    ap.add_argument("--pattern_uni2h", required=True)
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    conf_vits = load_conf(args.config_vits, "medical_ssl", args.n_token)
    conf_uni2h = load_conf(args.config_uni2h, "UNI2h", args.n_token)
    M = args.n_token

    def make_loader(conf):
        split_csv = os.path.join(conf.splits_dir, f'{args.split}.csv')
        feature_dir = conf.train_dir if args.split != 'test' else conf.test_dir
        dataset = PTClassificationDataset(split_csv=split_csv, feature_dir=feature_dir)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    loader_vits = make_loader(conf_vits)
    loader_uni2h = make_loader(conf_uni2h)

    vits_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_vits))}
    uni2h_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_uni2h))}
    common_seeds = sorted(set(vits_dirs) & set(uni2h_dirs))
    print(f"Спільні сіди (є і ViT-S/DINO, і UNI2h чекпоінт): {common_seeds}")

    rows = []
    pooled_flag_v, pooled_flag_u = [], []
    pooled_unflag_v, pooled_unflag_u = [], []
    for seed in common_seeds:
        vpath = os.path.join(vits_dirs[seed], args.ckpt_name)
        upath = os.path.join(uni2h_dirs[seed], args.ckpt_name)
        if not (os.path.exists(vpath) and os.path.exists(upath)):
            print(f"[skip] seed{seed}: немає чекпоінта")
            continue

        model_v = load_model(conf_vits, vpath, M)
        res_v = run_inference(model_v, loader_vits, want_disagreement=True)
        del model_v
        torch.cuda.empty_cache()

        model_u = load_model(conf_uni2h, upath, M)
        res_u = run_inference(model_u, loader_uni2h, want_disagreement=False)
        del model_u
        torch.cuda.empty_cache()

        common_ids = sorted(set(res_v) & set(res_u))
        if len(common_ids) < max(len(res_v), len(res_u)):
            print(f"  !!! seed{seed}: len(res_v)={len(res_v)} len(res_u)={len(res_u)} -- "
                  f"різна довжина test.csv між конфігами, перевір splits_dir")

        disagree = np.array([res_v[sid]['disagreement'] for sid in common_ids])
        true = np.array([res_v[sid]['true'] for sid in common_ids])
        pred_v = np.array([res_v[sid]['pred'] for sid in common_ids])
        pred_u = np.array([res_u[sid]['pred'] for sid in common_ids])

        flagged = disagree >= args.threshold
        acc_v_flag = (pred_v[flagged] == true[flagged]).mean() if flagged.sum() else np.nan
        acc_u_flag = (pred_u[flagged] == true[flagged]).mean() if flagged.sum() else np.nan
        acc_v_unflag = (pred_v[~flagged] == true[~flagged]).mean() if (~flagged).sum() else np.nan
        acc_u_unflag = (pred_u[~flagged] == true[~flagged]).mean() if (~flagged).sum() else np.nan

        d_flag = acc_u_flag - acc_v_flag
        d_unflag = acc_u_unflag - acc_v_unflag
        rows.append((seed, flagged.sum(), len(common_ids), acc_v_flag, acc_u_flag, d_flag,
                     acc_v_unflag, acc_u_unflag, d_unflag))
        print(f"seed{seed:2d}  n_flagged={flagged.sum():2d}/{len(common_ids)}  "
              f"flagged: vits={acc_v_flag:.3f} uni2h={acc_u_flag:.3f} Δ={d_flag:+.3f}   "
              f"unflagged: vits={acc_v_unflag:.3f} uni2h={acc_u_unflag:.3f} Δ={d_unflag:+.3f}")

        # Сирі per-slide correct/incorrect -- для об'єднаного McNemar на пулі
        # з усіх сідів (набагато потужніше за paired t-test по 10 середніх).
        correct_v = (pred_v == true).astype(int)
        correct_u = (pred_u == true).astype(int)
        pooled_flag_v.extend(correct_v[flagged].tolist())
        pooled_flag_u.extend(correct_u[flagged].tolist())
        pooled_unflag_v.extend(correct_v[~flagged].tolist())
        pooled_unflag_u.extend(correct_u[~flagged].tolist())

    rows = np.array([r[3:] for r in rows])  # [n_seeds, 6]
    d_flag_all = rows[:, 2]
    d_unflag_all = rows[:, 5]

    print(f"\n=== Усереднено по {len(rows)} сідах ===")
    print(f"Δacc на flagged (disagreement>=thr):   mean={np.nanmean(d_flag_all):+.4f} std={np.nanstd(d_flag_all, ddof=1):.4f}")
    print(f"Δacc на unflagged:                      mean={np.nanmean(d_unflag_all):+.4f} std={np.nanstd(d_unflag_all, ddof=1):.4f}")

    valid = ~(np.isnan(d_flag_all) | np.isnan(d_unflag_all))
    t, p = sstats.ttest_rel(d_flag_all[valid], d_unflag_all[valid])
    print(f"\nPaired t-test (Δacc_flagged vs Δacc_unflagged, по сідах): t={t:.3f} p={p:.4g}")
    print("(значущий позитивний t означає: UNI2h дійсно 'рятує' саме ті слайди,")
    print(" де ViT-S/DINO Tier1 сам собі не довіряв -- а не покращує все підряд)")

    # --- Об'єднаний McNemar's test: кожен (сід, слайд) -- окреме спостереження ---
    def mcnemar(correct_v, correct_u, label):
        correct_v = np.array(correct_v)
        correct_u = np.array(correct_u)
        n = len(correct_v)
        acc_v = correct_v.mean()
        acc_u = correct_u.mean()
        # b = vits вірно, uni2h невірно; c = vits невірно, uni2h вірно
        b = int(((correct_v == 1) & (correct_u == 0)).sum())
        c = int(((correct_v == 0) & (correct_u == 1)).sum())
        if b + c == 0:
            print(f"{label}: n={n} acc_vits={acc_v:.3f} acc_uni2h={acc_u:.3f} -- "
                  f"b=c=0, McNemar не визначений")
            return
        stat = (abs(b - c) - 1) ** 2 / (b + c)  # з поправкою на неперервність
        p_mcnemar = 1 - sstats.chi2.cdf(stat, df=1)
        print(f"{label}: n={n} acc_vits={acc_v:.3f} acc_uni2h={acc_u:.3f} "
              f"(vits-only-right={b}, uni2h-only-right={c})  McNemar p={p_mcnemar:.4g}")

    print("\n=== Об'єднаний McNemar's test (усі сіди разом, по слайдах) ===")
    mcnemar(pooled_flag_v, pooled_flag_u, "FLAGGED  ")
    mcnemar(pooled_unflag_v, pooled_unflag_u, "UNFLAGGED")


if __name__ == "__main__":
    main()
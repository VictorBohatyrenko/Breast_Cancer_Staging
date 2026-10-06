#!/usr/bin/env python
"""
End-to-end eval справжнього RoutedDTFDASMIL (architecture/routed_dtfd_asmil.py):
компонує вже натреновані дешеву (ViT-S/DINO) і дорогу (UNI2h) DTFD-ASMIL v2
моделі в один inference-каскад і рахує метрики НАПРЯМУ через forward()
модуля -- на відміну від uni2h_fixes_disagreement.py, який лише порівнював
готові предикшени offline. Це і є "продакшн"-версія перевіреної ідеї.

Запуск (з кореня Hi-MoE_MIL, з активованим venv; architecture/routed_dtfd_asmil.py
має лежати поруч з іншими файлами architecture/):
    python eval_routed_dtfd_asmil.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --cheap_ckpt ckpt/bracs_dtfd_v2_seed1/checkpoint-best.pth \
        --expensive_ckpt ckpt/bracs_dtfd_v2_uni2h_seed1/checkpoint-best.pth \
        --threshold 0.1 --split test
"""
import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from utils.utils import Struct
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2
from architecture.routed_dtfd_asmil import RoutedDTFDASMIL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PairedFeatureDataset(torch.utils.data.Dataset):
    """Читає ОДИН spliced CSV і завантажує .pt фічі того самого слайду з ДВОХ
    директорій (дешевий + дорогий backbone), зіставлені за slide_id з CSV --
    надійніше за зіставлення за порядковим індексом DataLoader-ів."""

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

        return {
            'feats_cheap': data_c['features'], 'coords_cheap': data_c['coords'].float(),
            'feats_expensive': data_e['features'], 'coords_expensive': data_e['coords'].float(),
            'label': label, 'slide_id': slide_id,
        }


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


def run_one_seed(conf_cheap, conf_exp, cheap_ckpt, expensive_ckpt, M, threshold, dataset):
    cheap_model = load_model(conf_cheap, cheap_ckpt, M)
    expensive_model = load_model(conf_exp, expensive_ckpt, M)
    routed = RoutedDTFDASMIL(cheap_model, expensive_model, threshold=threshold).to(device)
    routed.eval()

    routed_preds, cheap_preds, expensive_preds, trues, escalated = [], [], [], [], []
    with torch.no_grad():
        for item in dataset:
            feats_c = item['feats_cheap'].unsqueeze(0).to(device)
            feats_e = item['feats_expensive'].unsqueeze(0).to(device)
            coords_c = item['coords_cheap'].unsqueeze(0).to(device)
            coords_e = item['coords_expensive'].unsqueeze(0).to(device)

            slide_pred, used_expensive, disagreement = routed(
                feats_c, feats_e, coords_cheap=coords_c, coords_expensive=coords_e)
            cheap_only_pred, _, _ = cheap_model(feats_c, coords=coords_c)
            expensive_only_pred, _, _ = expensive_model(feats_e, coords=coords_e)

            routed_preds.append(int(slide_pred.argmax(dim=-1).item()))
            cheap_preds.append(int(cheap_only_pred.argmax(dim=-1).item()))
            expensive_preds.append(int(expensive_only_pred.argmax(dim=-1).item()))
            trues.append(item['label'])
            escalated.append(used_expensive)

    del cheap_model, expensive_model, routed
    torch.cuda.empty_cache()

    return (np.array(routed_preds), np.array(cheap_preds), np.array(expensive_preds),
            np.array(trues), np.array(escalated))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--cheap_ckpt", help="шлях ДО .pth файлу (single-seed режим)")
    ap.add_argument("--expensive_ckpt", help="шлях ДО .pth файлу (single-seed режим)")
    ap.add_argument("--pattern_cheap", help="glob ckpt_dir дешевих моделей, напр. 'ckpt/bracs_dtfd_v2_seed*' (multi-seed режим)")
    ap.add_argument("--pattern_expensive", help="glob ckpt_dir дорогих моделей, напр. 'ckpt/bracs_dtfd_v2_uni2h_seed*'")
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    args = ap.parse_args()

    conf_cheap = load_conf(args.config_cheap, "medical_ssl", args.n_token)
    conf_exp = load_conf(args.config_expensive, "UNI2h", args.n_token)
    M = args.n_token

    split_csv = os.path.join(conf_cheap.splits_dir, f'{args.split}.csv')
    feature_dir_cheap = conf_cheap.train_dir if args.split != 'test' else conf_cheap.test_dir
    feature_dir_exp = conf_exp.train_dir if args.split != 'test' else conf_exp.test_dir
    dataset = PairedFeatureDataset(split_csv, feature_dir_cheap, feature_dir_exp)

    if args.cheap_ckpt and args.expensive_ckpt:
        # single-seed режим (як раніше)
        routed_p, cheap_p, exp_p, trues, esc = run_one_seed(
            conf_cheap, conf_exp, args.cheap_ckpt, args.expensive_ckpt, M, args.threshold, dataset)
        print(f"Слайдів: {len(trues)}, ескальовано: {esc.sum()} ({100*esc.mean():.1f}%)")
        for name, preds in [("cheap-only", cheap_p), ("expensive-only", exp_p), ("ROUTED", routed_p)]:
            print(f"  {name:16s} acc={accuracy_score(trues, preds):.4f} "
                  f"f1_macro={f1_score(trues, preds, average='macro'):.4f}")
        return

    # multi-seed режим
    import re
    import glob
    from scipy import stats as sstats

    def extract_seed(dirname):
        m = re.search(r'seed(\d+)$', dirname)
        return int(m.group(1)) if m else None

    cheap_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_cheap))}
    exp_dirs = {extract_seed(os.path.basename(d)): d for d in sorted(glob.glob(args.pattern_expensive))}
    common_seeds = sorted(set(cheap_dirs) & set(exp_dirs))
    print(f"Спільні сіди: {common_seeds}")

    rows = []  # [acc_cheap, f1_cheap, acc_exp, f1_exp, acc_routed, f1_routed, pct_escalated]
    for seed in common_seeds:
        cheap_ckpt = os.path.join(cheap_dirs[seed], args.ckpt_name)
        exp_ckpt = os.path.join(exp_dirs[seed], args.ckpt_name)
        if not (os.path.exists(cheap_ckpt) and os.path.exists(exp_ckpt)):
            print(f"[skip] seed{seed}: немає чекпоінта")
            continue
        routed_p, cheap_p, exp_p, trues, esc = run_one_seed(
            conf_cheap, conf_exp, cheap_ckpt, exp_ckpt, M, args.threshold, dataset)

        acc_c, f1_c = accuracy_score(trues, cheap_p), f1_score(trues, cheap_p, average='macro')
        acc_e, f1_e = accuracy_score(trues, exp_p), f1_score(trues, exp_p, average='macro')
        acc_r, f1_r = accuracy_score(trues, routed_p), f1_score(trues, routed_p, average='macro')
        rows.append([acc_c, f1_c, acc_e, f1_e, acc_r, f1_r, 100 * esc.mean()])
        print(f"seed{seed:2d}  escalated={100*esc.mean():5.1f}%  "
              f"cheap: acc={acc_c:.4f} f1={f1_c:.4f}   "
              f"expensive: acc={acc_e:.4f} f1={f1_e:.4f}   "
              f"ROUTED: acc={acc_r:.4f} f1={f1_r:.4f}")

    rows = np.array(rows)
    names = ["acc_cheap", "f1_cheap", "acc_expensive", "f1_expensive", "acc_routed", "f1_routed", "pct_escalated"]
    means = rows.mean(axis=0)
    stds = rows.std(axis=0, ddof=1)

    print(f"\n=== Усереднено по {len(rows)} сідах ===")
    for name, m, s in zip(names, means, stds):
        print(f"  {name:16s} {m:.4f} ± {s:.4f}")

    print("\n--- Значущість: ROUTED vs cheap-only, ROUTED vs expensive-only (paired t-test + Wilcoxon) ---")
    for metric, r_col, base_col, label in [
        ("acc", 4, 0, "vs cheap-only"), ("f1", 5, 1, "vs cheap-only"),
        ("acc", 4, 2, "vs expensive-only"), ("f1", 5, 3, "vs expensive-only"),
    ]:
        t, p = sstats.ttest_rel(rows[:, r_col], rows[:, base_col])
        try:
            w, p_w = sstats.wilcoxon(rows[:, r_col], rows[:, base_col])
        except ValueError:
            p_w = float('nan')
        delta = rows[:, r_col].mean() - rows[:, base_col].mean()
        print(f"  {metric} ROUTED {label:20s} Δ={delta:+.4f}  paired-t p={p:.4g}  wilcoxon p={p_w:.4g}")


if __name__ == "__main__":
    main()
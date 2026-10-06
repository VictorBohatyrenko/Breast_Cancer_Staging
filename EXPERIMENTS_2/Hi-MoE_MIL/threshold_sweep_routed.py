#!/usr/bin/env python
"""
Threshold sweep для RoutedDTFDASMIL: рахує forward pass ОДИН РАЗ на сід
(cheap_pred, expensive_pred, tier1_disagreement, true), а поріг ескалації
підбирає потім у numpy -- без повторних forward passes. Значно швидше за
повторний виклик eval_routed_dtfd_asmil.py для кожного порогу окремо.

Запуск:
    python threshold_sweep_routed.py \
        --config_cheap config/camelyon17_vitsdino_config_30ep.yml \
        --config_expensive config/camelyon17_uni2h_config.yml \
        --pattern_cheap "ckpt/c17_dtfd_v2_20ep_seed*" \
        --pattern_expensive "ckpt/c17_dtfd_v2_uni2h_seed*" \
        --ckpt_name checkpoint-best.pth \
        --thresholds 0.02,0.05,0.075,0.1,0.15,0.2,0.3
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
                'label': label}


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


def disagreement_of(tier1_slide_preds):
    probs = torch.softmax(tier1_slide_preds, dim=-1)
    argmax = probs.argmax(dim=-1)
    mode = torch.mode(argmax).values
    return (argmax != mode).float().mean().item()


def compute_seed_data(conf_cheap, conf_exp, cheap_ckpt, expensive_ckpt, M, dataset):
    """Один forward pass на слайд для обох моделей; повертає сирі масиви,
    з яких потім будь-який поріг рахується без повторного inference."""
    cheap_model = load_model(conf_cheap, cheap_ckpt, M)
    expensive_model = load_model(conf_exp, expensive_ckpt, M)

    disagree, cheap_pred, exp_pred, trues = [], [], [], []
    with torch.no_grad():
        for item in dataset:
            feats_c = item['feats_cheap'].unsqueeze(0).to(device)
            feats_e = item['feats_expensive'].unsqueeze(0).to(device)
            coords_c = item['coords_cheap'].unsqueeze(0).to(device)
            coords_e = item['coords_expensive'].unsqueeze(0).to(device)

            cp, t1preds, _ = cheap_model(feats_c, coords=coords_c)
            ep, _, _ = expensive_model(feats_e, coords=coords_e)

            disagree.append(disagreement_of(t1preds))
            cheap_pred.append(int(cp.argmax(dim=-1).item()))
            exp_pred.append(int(ep.argmax(dim=-1).item()))
            trues.append(item['label'])

    del cheap_model, expensive_model
    torch.cuda.empty_cache()
    return (np.array(disagree), np.array(cheap_pred), np.array(exp_pred), np.array(trues))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--pattern_cheap", required=True)
    ap.add_argument("--pattern_expensive", required=True)
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--thresholds", default="0.02,0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
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
        print(f"Прогін seed{seed} ...")
        per_seed_data[seed] = compute_seed_data(conf_cheap, conf_exp, cheap_ckpt, exp_ckpt, M, dataset)

    print(f"\n{'thr':>6s}  {'%esc':>6s}  {'acc_routed':>11s}  {'f1_routed':>10s}  "
          f"{'p(vs cheap)':>12s}  {'p(vs exp)':>10s}")
    for thr in thresholds:
        acc_list, f1_list, esc_list = [], [], []
        acc_cheap_list, f1_cheap_list = [], []
        acc_exp_list, f1_exp_list = [], []
        for seed, (disagree, cheap_pred, exp_pred, trues) in per_seed_data.items():
            flagged = disagree >= thr
            routed_pred = np.where(flagged, exp_pred, cheap_pred)
            acc_list.append(accuracy_score(trues, routed_pred))
            f1_list.append(f1_score(trues, routed_pred, average='macro'))
            esc_list.append(100 * flagged.mean())
            acc_cheap_list.append(accuracy_score(trues, cheap_pred))
            f1_cheap_list.append(f1_score(trues, cheap_pred, average='macro'))
            acc_exp_list.append(accuracy_score(trues, exp_pred))
            f1_exp_list.append(f1_score(trues, exp_pred, average='macro'))

        acc_list, f1_list = np.array(acc_list), np.array(f1_list)
        acc_cheap_list, acc_exp_list = np.array(acc_cheap_list), np.array(acc_exp_list)

        _, p_vs_cheap = sstats.ttest_rel(acc_list, acc_cheap_list)
        _, p_vs_exp = sstats.ttest_rel(acc_list, acc_exp_list)

        print(f"{thr:6.3f}  {np.mean(esc_list):6.1f}  "
              f"{acc_list.mean():.4f}±{acc_list.std(ddof=1):.3f}  "
              f"{f1_list.mean():.4f}±{f1_list.std(ddof=1):.3f}  "
              f"{p_vs_cheap:12.4g}  {p_vs_exp:10.4g}")


if __name__ == "__main__":
    main()

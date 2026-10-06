#!/usr/bin/env python
"""
Перевіряє ідею "consult on disagreement": головна модель (--main_seed) видає
вердикт сама; якщо її Tier1 psevdo-bags (M=8) не погоджуються між собою
(tier1_disagreement > поріг) -- скликаємо "консиліум" з ІНШИХ вже натренованих
сідів (--pattern) і берем majority vote з усіх M_SEEDS моделей для ЦЬОГО
слайду. Порівнюємо 3 режими:
  (a) baseline       -- тільки головна модель, завжди
  (b) routed-ensemble -- (a), але на "непевних" слайдах -- majority vote з усіх сідів
  (c) full-ensemble  -- majority vote з усіх сідів, на КОЖНОМУ слайді (дорого)

Порахує accuracy/F1 для (a)/(b)/(c) + скільки % слайдів реально пішло на
"консиліум" (тобто скільки додаткового компуту насправді треба для routed).

Запуск (з кореня Hi-MoE_MIL, з активованим venv):
    python disagreement_routing.py \
        --config config/bracs_vits_dino_config_100ep.yml \
        --pretrain medical_ssl \
        --pattern "ckpt/bracs_dtfd_v2_seed*" \
        --ckpt_name checkpoint-best.pth \
        --split test \
        --threshold 0.1

Тепер прогін ОДИН РАЗ рахує предикшени/disagreement для всіх 10 сідів, а тоді
цикл leave-one-out перебирає, який сід "головний" (по черзі кожен), і рахує
(a)/(b)/(c) для кожного -- це прибирає ефект "нам пощастило з конкретним
сідом" на маленькому test-сеті (n=86 -- одна помилка = ~1.2% acc).
"""
import argparse
import glob
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

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
    elif pretrain == 'UNI':
        conf.D_feat, conf.D_inner = 1024, 512
    elif pretrain == 'GigaPath':
        conf.D_feat, conf.D_inner = 1536, 768
    return conf


def load_model(conf, ckpt_path, M):
    model = DTFD_ASMIL_v2(conf, n_token=conf.n_token, M=M).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model'] if 'model' in state else state
    model.load_state_dict(sd)
    model.eval()
    return model


def run_all_slides(model, loader):
    """Повертає preds [n_slides], probs [n_slides, n_class], disagreement [n_slides], true [n_slides]."""
    preds, probs_all, disagree_all, trues = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            image_patches, label = batch[0].to(device), batch[1]
            final_pred, tier1_slide_preds, _ = model(image_patches)
            probs = torch.softmax(final_pred, dim=-1).squeeze(0).cpu().numpy()
            preds.append(int(probs.argmax()))
            probs_all.append(probs)
            trues.append(int(label.item() if torch.is_tensor(label) else label))

            t1_probs = torch.softmax(tier1_slide_preds, dim=-1)
            t1_argmax = t1_probs.argmax(dim=-1)
            mode_class = torch.mode(t1_argmax).values
            disagree_all.append((t1_argmax != mode_class).float().mean().item())
    return np.array(preds), np.stack(probs_all), np.array(disagree_all), np.array(trues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pretrain", default="medical_ssl")
    ap.add_argument("--pattern", required=True, help="glob усіх ckpt_dir (усі сіди -- 'консиліум')")
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.1,
                     help="tier1_disagreement >= поріг -> скликати консиліум")
    args = ap.parse_args()

    conf = load_conf(args.config, args.pretrain, args.n_token)
    M = args.n_token

    split_csv = os.path.join(conf.splits_dir, f'{args.split}.csv')
    feature_dir = conf.train_dir if args.split != 'test' else conf.test_dir
    dataset = PTClassificationDataset(split_csv=split_csv, feature_dir=feature_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_dirs = sorted(glob.glob(args.pattern))
    print(f"Знайдено {len(ckpt_dirs)} сідів для консиліуму: {[os.path.basename(d) for d in ckpt_dirs]}")

    all_preds, all_probs, all_disagree = [], [], []
    seed_names = []
    for i, d in enumerate(ckpt_dirs):
        ckpt_path = os.path.join(d, args.ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[skip] немає {ckpt_path}")
            continue
        print(f"Прогін {d} ...")
        model = load_model(conf, ckpt_path, M)
        preds, probs, disagree, trues = run_all_slides(model, loader)
        all_preds.append(preds)
        all_probs.append(probs)
        all_disagree.append(disagree)
        seed_names.append(os.path.basename(d))

    all_preds = np.stack(all_preds, axis=0)      # [n_seeds, n_slides]
    all_disagree = np.stack(all_disagree, axis=0)  # [n_seeds, n_slides]
    n_seeds, n_slides = all_preds.shape

    def majority_vote(preds_2d):
        from scipy import stats as sstats
        return sstats.mode(preds_2d, axis=0, keepdims=False).mode

    full_ensemble_preds = majority_vote(all_preds)
    acc_c = accuracy_score(trues, full_ensemble_preds)
    f1_c = f1_score(trues, full_ensemble_preds, average='macro')

    # Leave-one-out: кожен сід по черзі -- "головна" модель, решта 9 -- консиліум,
    # щоб виключити ефект "нам пощастило з головним сідом" (n=86 -- це один
    # хибний слайд = ~1.2% зсуву acc, тому одна точка нічого не доводить).
    print(f"\nПоріг disagreement = {args.threshold}")
    rows = []
    for main_idx in range(n_seeds):
        baseline_preds = all_preds[main_idx]
        main_disagree = all_disagree[main_idx]
        flagged = main_disagree >= args.threshold

        routed_preds = baseline_preds.copy()
        routed_preds[flagged] = full_ensemble_preds[flagged]

        acc_a = accuracy_score(trues, baseline_preds)
        f1_a = f1_score(trues, baseline_preds, average='macro')
        acc_b = accuracy_score(trues, routed_preds)
        f1_b = f1_score(trues, routed_preds, average='macro')
        rows.append((seed_names[main_idx], flagged.mean(), acc_a, f1_a, acc_b, f1_b))
        print(f"  {seed_names[main_idx]:22s} flagged={100*flagged.mean():5.1f}%  "
              f"(a) acc={acc_a:.4f} f1={f1_a:.4f}   (b) acc={acc_b:.4f} f1={f1_b:.4f}   "
              f"Δacc={acc_b-acc_a:+.4f} Δf1={f1_b-f1_a:+.4f}")

    rows_arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    mean_flag, mean_acc_a, mean_f1_a, mean_acc_b, mean_f1_b = rows_arr.mean(axis=0)
    std_acc_a, std_f1_a, std_acc_b, std_f1_b = rows_arr[:, 1:].std(axis=0, ddof=1)

    print("\n=== Усереднено по 10 leave-one-out прогонах (кожен сід як головний) ===")
    print(f"(a) baseline (1 модель, середнє):                  acc={mean_acc_a:.4f}±{std_acc_a:.4f}  "
          f"f1={mean_f1_a:.4f}±{std_f1_a:.4f}")
    print(f"(b) routed-ensemble (консиліум на ~{100*mean_flag:.0f}% слайдів): acc={mean_acc_b:.4f}±{std_acc_b:.4f}  "
          f"f1={mean_f1_b:.4f}±{std_f1_b:.4f}")
    print(f"(c) full-ensemble (усі 10 завжди, x10 компуту):    acc={acc_c:.4f}  f1={f1_c:.4f}")

    from scipy.stats import ttest_rel
    t_acc, p_acc = ttest_rel(rows_arr[:, 3], rows_arr[:, 1])
    t_f1, p_f1 = ttest_rel(rows_arr[:, 4], rows_arr[:, 2])
    print(f"\nPaired t-test (b) vs (a): Δacc p={p_acc:.4g}, Δf1 p={p_f1:.4g}")


if __name__ == "__main__":
    main()
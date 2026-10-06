#!/usr/bin/env python
"""
Швидка перевірка двох питань перед тим, як вкладати ще час у Tier1.5:

 (1) СТЕЛЯ (headroom). Скільки взагалі можна виграти ескалацією? На тестових
     слайдах для кожного сіда рахуємо:
        acc_cheap, acc_exp,
        rescue = частка слайдів, де cheap ПОМИЛЯЄТЬСЯ, а expensive ПРАВИЙ
                 (це максимум, який може "врятувати" ідеальна маршрутизація),
        harm   = частка слайдів, де cheap правий, а expensive ПОМИЛЯЄТЬСЯ,
        oracle = acc ідеального маршрутизатора (бере кращу з двох моделей).
     Якщо rescue мала (кілька %), Tier1.5 просто немає що виправляти.

 (2) ПАРНИЙ ТЕСТ Tier1.5 vs cheap-only на ТИХ САМИХ сідах (за наявності
     логів тренування Tier1.5: парсимо рядки test_acc=... test_f1=...).

Cheap/expensive ітерації рахуються з n_repeats випадковими розбиттями bags
(як в інших скриптах проєкту), метрика усереднюється по повтореннях --
тому cheap-числа тут можуть трохи відрізнятись від сорокових baseline-ів з
логів тренування.

Запуск (з кореня Hi-MoE_MIL, де лежить validate_bag_level_routing.py):
    python headroom_and_paired.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --pattern_cheap 'ckpt/bracs_dtfd_v2_seed*' \
        --pattern_expensive 'ckpt/bracs_dtfd_v2_uni2h_seed*' \
        --tier1_5_logs 'ckpt/bracs_tier1_5_seed*.log'
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
from scipy import stats as sstats
from sklearn.metrics import f1_score

from validate_bag_level_routing import PairedFeatureDataset, load_conf, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_seed(path):
    m = re.search(r'seed(\d+)', os.path.basename(path.rstrip('/')))
    return int(m.group(1)) if m else None


@torch.no_grad()
def predict(model, dataset, which, n_repeats):
    """Повертає preds [n_repeats, n_slides], labels [n_slides]."""
    preds = np.zeros((n_repeats, len(dataset)), dtype=int)
    labels = np.zeros(len(dataset), dtype=int)
    for i in range(len(dataset)):
        item = dataset[i]
        feats = item[f'feats_{which}'].unsqueeze(0).to(device)
        coords = item[f'coords_{which}'].unsqueeze(0).to(device)
        labels[i] = item['label']
        for r in range(n_repeats):
            out, _, _ = model(feats, coords=coords)
            preds[r, i] = int(out.argmax(dim=-1).item())
    return preds, labels


def parse_logs(pattern):
    res = {}
    for path in sorted(glob.glob(pattern)):
        seed = extract_seed(path)
        txt = open(path, encoding='utf-8', errors='ignore').read()
        m = re.findall(r'test_acc=([0-9.]+)\s+test_f1=([0-9.]+)', txt)
        if seed is not None and m:
            res[seed] = (float(m[-1][0]), float(m[-1][1]))
    return res


def paired_report(name, a, b, label_a, label_b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = sstats.ttest_rel(a, b)
    print(f"  {name}: {label_a}={a.mean():.4f}±{a.std(ddof=1):.4f}  "
          f"{label_b}={b.mean():.4f}±{b.std(ddof=1):.4f}  "
          f"Δ={d.mean():+.4f}  (сідів з Δ>0: {(d > 0).sum()}/{len(d)})  "
          f"парний t-тест p={p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--pattern_cheap", required=True)
    ap.add_argument("--pattern_expensive", required=True)
    ap.add_argument("--tier1_5_logs", default=None,
                    help="glob логів тренування Tier1.5, напр. 'ckpt/bracs_tier1_5_seed*.log' "
                         "(потрібні рядки 'test_acc=... test_f1=...')")
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--n_repeats", type=int, default=5)
    args = ap.parse_args()

    conf_c = load_conf(args.config_cheap, "medical_ssl", args.n_token)
    conf_e = load_conf(args.config_expensive, "UNI2h", args.n_token)
    M = args.n_token
    split_csv = os.path.join(conf_c.splits_dir, f'{args.split}.csv')
    fd_c = conf_c.test_dir if args.split == 'test' else conf_c.train_dir
    fd_e = conf_e.test_dir if args.split == 'test' else conf_e.train_dir
    ds = PairedFeatureDataset(split_csv, fd_c, fd_e)

    cd = {extract_seed(d): d for d in glob.glob(args.pattern_cheap)}
    ed = {extract_seed(d): d for d in glob.glob(args.pattern_expensive)}
    seeds = sorted(set(cd) & set(ed))
    print(f"Спільні сіди: {seeds}; тест-слайдів: {len(ds)}; n_repeats={args.n_repeats}\n")

    rows = {}
    for s in seeds:
        cp = os.path.join(cd[s], args.ckpt_name)
        ep = os.path.join(ed[s], args.ckpt_name)
        if not (os.path.exists(cp) and os.path.exists(ep)):
            print(f"[skip] seed{s}")
            continue
        mc = load_model(conf_c, cp, M)
        pc, y = predict(mc, ds, 'cheap', args.n_repeats)
        del mc
        me = load_model(conf_e, ep, M)
        pe, _ = predict(me, ds, 'expensive', args.n_repeats)
        del me
        cc, ce = (pc == y), (pe == y)  # [R, N]
        rows[s] = dict(
            acc_c=cc.mean(), acc_e=ce.mean(),
            rescue=(~cc & ce).mean(), harm=(cc & ~ce).mean(), oracle=(cc | ce).mean(),
            f1_c=np.mean([f1_score(y, pc[r], average='macro') for r in range(args.n_repeats)]),
            f1_e=np.mean([f1_score(y, pe[r], average='macro') for r in range(args.n_repeats)]),
        )
        r = rows[s]
        print(f"seed{s:>2}: acc cheap={r['acc_c']:.4f} exp={r['acc_e']:.4f} | "
              f"rescue={r['rescue']:.4f} harm={r['harm']:.4f} oracle={r['oracle']:.4f}")

    S = sorted(rows)
    g = lambda k: np.array([rows[s][k] for s in S])
    print("\n=== (1) СТЕЛЯ (середнє по сідах) ===")
    print(f"  acc cheap      = {g('acc_c').mean():.4f}")
    print(f"  acc expensive  = {g('acc_e').mean():.4f}")
    print(f"  oracle (ідеальна маршрутизація) = {g('oracle').mean():.4f}  "
          f"-> макс. приріст над cheap = {(g('oracle') - g('acc_c')).mean():+.4f}")
    print(f"  rescue (cheap ✗, exp ✓) = {g('rescue').mean():.4f}   "
          f"harm (cheap ✓, exp ✗) = {g('harm').mean():.4f}")
    paired_report("expensive vs cheap, acc", g('acc_e'), g('acc_c'), 'exp', 'cheap')
    paired_report("expensive vs cheap, F1 ", g('f1_e'), g('f1_c'), 'exp', 'cheap')

    if args.tier1_5_logs:
        t15 = parse_logs(args.tier1_5_logs)
        common = [s for s in S if s in t15]
        print(f"\n=== (2) Tier1.5 vs cheap-only, парні по сідах {common} ===")
        if len(common) >= 3:
            ta = [t15[s][0] for s in common]
            tf = [t15[s][1] for s in common]
            paired_report("acc", ta, [rows[s]['acc_c'] for s in common], 'tier1.5', 'cheap')
            paired_report("F1 ", tf, [rows[s]['f1_c'] for s in common], 'tier1.5', 'cheap')
            paired_report("acc vs expensive", ta, [rows[s]['acc_e'] for s in common], 'tier1.5', 'exp')
        else:
            print("  Замало логів з рядками test_acc/test_f1 (потрібно >=3 сідів).")


if __name__ == "__main__":
    main()

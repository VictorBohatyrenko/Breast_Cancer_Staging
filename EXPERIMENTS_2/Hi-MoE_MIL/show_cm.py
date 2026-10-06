#!/usr/bin/env python
"""
Виводить confusion matrix по кожному сіду + агреговану (сума і row-normalized)
для набору ckpt_dir з файлами test_confusion_matrix_best.npy (або _final.npy).

Приклад використання (запускати на сервері, там де лежать ckpt/):
    python show_cm.py --pattern "ckpt/dtfd_asmil_v2_bracs_seed*" --which best

Якщо не знаєш точний ckpt_dir-паттерн — спочатку подивись:
    ls -d ckpt/*dtfd*bracs* ckpt/*v2*bracs*
"""
import argparse
import glob
import os
import numpy as np

CLASS_NAMES = ["benign", "atypical", "malignant"]


def load_cm(ckpt_dir, which):
    fname = f"test_confusion_matrix_{which}.npy"
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        return None
    return np.load(path)


def print_cm(cm, title, class_names):
    print(title)
    n = cm.shape[0]
    names = class_names[:n]
    header = "        " + "  ".join(f"{c:>10s}" for c in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:>8s} " + "  ".join(f"{v:>10.0f}" for v in row))
    # per-class recall (row-normalized)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = cm / row_sums
    print("  (row-normalized, = per-class recall)")
    for i, row in enumerate(norm):
        print(f"{names[i]:>8s} " + "  ".join(f"{v:>10.3f}" for v in row))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="glob для ckpt_dir, напр. 'ckpt/dtfd_asmil_v2_bracs_seed*'")
    ap.add_argument("--which", default="best", choices=["best", "final"])
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.pattern))
    if not dirs:
        print(f"Нічого не знайдено за паттерном {args.pattern}")
        return

    cms = []
    for d in dirs:
        cm = load_cm(d, args.which)
        if cm is None:
            print(f"[skip] {d}: немає test_confusion_matrix_{args.which}.npy")
            continue
        cms.append(cm)
        print_cm(cm, f"=== {os.path.basename(d)} ===", CLASS_NAMES)

    if not cms:
        return

    stacked = np.stack(cms, axis=0)  # [n_seeds, n_class, n_class]
    summed = stacked.sum(axis=0)
    print_cm(summed, f"=== СУМА по {len(cms)} сідах ===", CLASS_NAMES)

    mean_norm = np.stack([cm / cm.sum(axis=1, keepdims=True).clip(min=1) for cm in cms], axis=0).mean(axis=0)
    print("=== Середній per-class recall по сідах (mean ± std) ===")
    std_norm = np.stack([cm / cm.sum(axis=1, keepdims=True).clip(min=1) for cm in cms], axis=0).std(axis=0)
    n = mean_norm.shape[0]
    names = CLASS_NAMES[:n]
    print("        " + "  ".join(f"{c:>14s}" for c in names))
    for i in range(n):
        row_str = "  ".join(f"{mean_norm[i,j]:.3f}±{std_norm[i,j]:.3f}" for j in range(n))
        print(f"{names[i]:>8s} " + row_str)


if __name__ == "__main__":
    main()

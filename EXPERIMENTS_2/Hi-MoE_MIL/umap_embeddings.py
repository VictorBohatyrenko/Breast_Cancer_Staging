#!/usr/bin/env python
"""
Витягує slide-level ембедінги (pooled-вектор Tier2, ПЕРЕД класифікатором,
розмірність D_inner) з навченого чекпоінта DTFD-ASMIL v2 і будує UMAP-проекцію,
пофарбовану за true-класом (і окремо -- за correct/incorrect).

Запуск (з кореня Hi-MoE_MIL, з активованим venv):
    pip install umap-learn --break-system-packages   # якщо ще нема

    python umap_embeddings.py \
        --config config/bracs_vits_dino_config_100ep.yml \
        --pretrain medical_ssl \
        --ckpt_dir ckpt/bracs_dtfd_v2_seed1 \
        --ckpt_name checkpoint_29.pth \
        --split test \
        --out umap_bracs_dtfd_v2_seed1.png

--ckpt_name: назва файлу чекпоінта в ckpt_dir. У v2-скрипті зберігається
checkpoint-best.pth (якщо val покращився) та checkpoint_{epoch}.pth (щоепохи) --
подивись `ls ckpt/bracs_dtfd_v2_seed1/` і вибери потрібний (найчастіше
останню епоху, бо test_confusion_matrix_final.npy теж рахувався на ній).

Можна також об'єднати кілька сідів на одному графіку через --pattern замість
--ckpt_dir (тоді для кожного сіда береться той самий --ckpt_name), щоб
подивитись, чи кластерна структура стабільна між сідами.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils import Struct
from pt_dataset import PTClassificationDataset
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2

DATASET_CLASS_NAMES = {
    'camelyon17': ['negative', 'itc', 'micro', 'macro'],
    'camelyon16': ['normal', 'tumor'],
    'bracs': ['benign', 'atypical', 'malignant'],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_conf(config_path, pretrain, n_token=8, M=None, tier2_n_heads=4, tier2_D_inner=None):
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
    return conf, (M if M is not None else n_token), tier2_n_heads, tier2_D_inner


def extract_embeddings(conf, M, tier2_n_heads, tier2_D_inner, ckpt_path, loader, class_names):
    model = DTFD_ASMIL_v2(conf, n_token=conf.n_token, M=M,
                           tier2_n_heads=tier2_n_heads, tier2_D_inner=tier2_D_inner).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model'] if 'model' in state else state
    model.load_state_dict(sd)
    model.eval()

    # Hook на classifier Tier2: його INPUT -- це саме pooled slide-embedding
    # (див. architecture/lsa_transformer_tier2.py: `pooled = ...; slide_pred =
    # self.classifier(pooled)`). Простіше і надійніше за копіювання forward().
    captured = {}
    def hook(module, inp, out):
        captured['pooled'] = inp[0].detach().cpu()
    handle = model.tier2.classifier.register_forward_hook(hook)

    embeddings, true_labels, pred_labels, slide_ids, n_tiles_list = [], [], [], [], []
    tier1_disagree_list, tier1_entropy_list = [], []
    with torch.no_grad():
        for batch in loader:
            image_patches, label, slide_id = batch[0], batch[1], batch[2] if len(batch) > 2 else None
            image_patches = image_patches.to(device)
            final_pred, tier1_slide_preds, _ = model(image_patches)
            embeddings.append(captured['pooled'].squeeze(0).numpy())
            true_labels.append(int(label.item() if torch.is_tensor(label) else label))
            pred_labels.append(int(final_pred.argmax(dim=-1).item()))
            slide_ids.append(slide_id[0] if slide_id is not None else len(slide_ids))
            n_tiles_list.append(image_patches.shape[1])  # AGSD/diag: розмір WSI (кількість тайлів)

            # Гіпотеза 2: наскільки M=8 псевдо-bags Tier1 не погоджуються між
            # собою щодо класу цього слайду -- частка bags, чий argmax != мода
            # (найчастіший клас серед bags), плюс середня softmax-ентропія
            # окремого bag'а (низька ентропія = bag впевнений сам по собі).
            t1_probs = torch.softmax(tier1_slide_preds, dim=-1)  # [M, n_class]
            t1_argmax = t1_probs.argmax(dim=-1)  # [M]
            mode_class = torch.mode(t1_argmax).values
            disagreement = (t1_argmax != mode_class).float().mean().item()
            entropy = (-(t1_probs * torch.log(t1_probs.clamp_min(1e-8))).sum(dim=-1)).mean().item()
            tier1_disagree_list.append(disagreement)
            tier1_entropy_list.append(entropy)

    handle.remove()
    return (np.stack(embeddings), np.array(true_labels), np.array(pred_labels), slide_ids,
            np.array(n_tiles_list), np.array(tier1_disagree_list), np.array(tier1_entropy_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pretrain", default="medical_ssl")
    ap.add_argument("--ckpt_dir", default=None, help="один ckpt_dir")
    ap.add_argument("--pattern", default=None, help="glob кількох ckpt_dir (об'єднати всі сіди)")
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--out", default="umap_embeddings.png")
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    args = ap.parse_args()

    conf, M, tier2_n_heads, tier2_D_inner = load_conf(args.config, args.pretrain, args.n_token)
    class_names = DATASET_CLASS_NAMES.get(conf.dataset, [f'class{i}' for i in range(conf.n_class)])

    split_csv = os.path.join(conf.splits_dir, f'{args.split}.csv')
    feature_dir = conf.train_dir if args.split != 'test' else conf.test_dir
    dataset = PTClassificationDataset(split_csv=split_csv, feature_dir=feature_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_dirs = sorted(glob.glob(args.pattern)) if args.pattern else [args.ckpt_dir]
    all_emb, all_true, all_pred, all_seed, all_ntiles = [], [], [], [], []
    all_disagree, all_entropy = [], []
    for d in ckpt_dirs:
        ckpt_path = os.path.join(d, args.ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[skip] немає {ckpt_path}")
            continue
        print(f"Обробляю {d} ...")
        emb, true_l, pred_l, slide_ids, n_tiles, disagree, entropy = extract_embeddings(
            conf, M, tier2_n_heads, tier2_D_inner, ckpt_path, loader, class_names)
        all_emb.append(emb)
        all_true.append(true_l)
        all_pred.append(pred_l)
        all_ntiles.append(n_tiles)
        all_disagree.append(disagree)
        all_entropy.append(entropy)
        all_seed.extend([os.path.basename(d)] * len(true_l))

    embeddings = np.concatenate(all_emb, axis=0)
    true_labels = np.concatenate(all_true, axis=0)
    pred_labels = np.concatenate(all_pred, axis=0)
    n_tiles_all = np.concatenate(all_ntiles, axis=0)
    disagree_all = np.concatenate(all_disagree, axis=0)
    entropy_all = np.concatenate(all_entropy, axis=0)

    print(f"Загалом {embeddings.shape[0]} ембедінгів, розмірність {embeddings.shape[1]}")

    import umap
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                         metric="cosine", random_state=42)
    proj = reducer.fit_transform(embeddings)

    correct = (true_labels == pred_labels)

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    ax = axes[0]
    for i, cname in enumerate(class_names):
        mask = true_labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=18, alpha=0.75,
                   color=colors[i], label=cname)
    ax.set_title(f"UMAP ({args.split}), колір = справжній клас")
    ax.legend()

    ax = axes[1]
    ax.scatter(proj[correct, 0], proj[correct, 1], s=18, alpha=0.6, color="tab:green", label="correct")
    ax.scatter(proj[~correct, 0], proj[~correct, 1], s=24, alpha=0.9, color="tab:red",
               marker="x", label="misclassified")
    ax.set_title("UMAP, correct vs misclassified")
    ax.legend()

    ax = axes[2]
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=18, alpha=0.8, c=np.log10(n_tiles_all), cmap="viridis")
    ax.set_title("UMAP, колір = log10(n_tiles) — розмір WSI")
    plt.colorbar(sc, ax=ax, label="log10(n_tiles)")

    ax = axes[3]
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=18, alpha=0.8, c=disagree_all, cmap="magma")
    ax.set_title("UMAP, колір = Tier1 disagreement (частка з M=8 bags != мода)")
    plt.colorbar(sc, ax=ax, label="disagreement")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Збережено: {args.out}")

    # Кількісна перевірка: формально кластеризуємо 2D UMAP-проекцію на 3
    # кластери (KMeans, вони й так добре розділені на око) і дивимось, який
    # фактор -- true_label, n_tiles, tier1-розбіжність чи tier1-ентропія --
    # пояснює приналежність до кластера краще (chi2 / ANOVA-подібні тести).
    from scipy import stats as sstats
    from sklearn.cluster import KMeans

    r0, p0 = sstats.pearsonr(np.log10(n_tiles_all), proj[:, 0])
    r1, p1 = sstats.pearsonr(np.log10(n_tiles_all), proj[:, 1])
    print(f"Кореляція log10(n_tiles) з UMAP-x: r={r0:.3f} p={p0:.4g}")
    print(f"Кореляція log10(n_tiles) з UMAP-y: r={r1:.3f} p={p1:.4g}")

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(proj)
    cluster_id = km.labels_

    print("\n--- Формальна перевірка факторів по 3 UMAP-кластерах (KMeans) ---")
    # chi2: чи true_label залежить від cluster_id
    ct_class = pd.crosstab(cluster_id, true_labels)
    chi2, p_chi2, _, _ = sstats.chi2_contingency(ct_class)
    print(f"true_label vs cluster: chi2 p={p_chi2:.4g}\n{ct_class}")

    # ANOVA/Kruskal: чи n_tiles / disagreement / entropy різняться між кластерами
    for name, values in [("log10(n_tiles)", np.log10(n_tiles_all)),
                          ("tier1_disagreement", disagree_all),
                          ("tier1_entropy", entropy_all)]:
        groups = [values[cluster_id == c] for c in range(3)]
        h, p_kw = sstats.kruskal(*groups)
        means = [f"{g.mean():.3f}" for g in groups]
        print(f"{name} по кластерах (Kruskal-Wallis p={p_kw:.4g}): means={means}")

    np.savez(args.out.replace(".png", "_data.npz"),
             embeddings=embeddings, proj=proj, true_labels=true_labels, pred_labels=pred_labels,
             n_tiles=n_tiles_all, tier1_disagreement=disagree_all, tier1_entropy=entropy_all,
             cluster_id=cluster_id)
    print(f"Сирі дані: {args.out.replace('.png', '_data.npz')}")


if __name__ == "__main__":
    main()
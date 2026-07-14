"""
Full CAMELYON17 pipeline: UNI2-h feature extraction -> CLAM-SB MIL training
(with embedding augmentation + anti-overfitting regularization) -> pN submission.csv.

Stages are independent and resumable (encoder skips existing .pt, run only what you need):
    python clam_pipeline.py --encode-train
    python clam_pipeline.py --encode-test
    python clam_pipeline.py --train
    python clam_pipeline.py --infer
    python clam_pipeline.py --eval-train --eval-test
    python clam_pipeline.py --encode-train --encode-test --train --infer
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import timm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
CFG = {
    "seed": 42,
    "val_size": 0.15,
    "test_size": 0.15,
    # --- encoder (reads the .h5 tiles produced by tiling_and_filtration.py) ---
    "train_h5_dir": "/data/bohatyrenko1/patches_jpeg_level1_training",
    "test_h5_dir": "/data/bohatyrenko1/patches_jpeg_level1",
    "wandb_project": "camelyon17-mil",
    "wandb_run_name": None,
    "train_features_root": "/data/bohatyrenko1/features_uni2h_training",  # flat <slide>.pt
    "test_features_dir": "/data/bohatyrenko1/features_uni2h_test",
    "labels_csv": "/home/root_server/bohatyrenko/CAMELYON/stage_labels.csv",
    "test_labels_xlsx": "/data/bohatyrenko1/experiments/Camelyon+(4-classes).xlsx",  # ground truth for patient_100-199, not in labels_csv
    "test_patient_lo": 100,
    "test_patient_hi": 199,
    "cache_dir": "/home/root_server/bohatyrenko/CAMELYON/.cache/uni2_h",
    "k_max_patches": 2048,       # random subsample if an .h5 has more tiles than this
    "encode_batch_size": 128,
    "encode_num_workers": 12,
    "encode_prefetch": 4,
    # --- training ---
    "save_dir": "/home/root_server/bohatyrenko/experiments/mil_checkpoints",
    "submission_csv": "/home/root_server/bohatyrenko/experiments/submission.csv",
    "d_emb": 1536,
    "hid": 256,
    "attn_dim": 128,
    "dropout": 0.3,
    "feat_dropout": 0.1,        # zero-out random embedding dims (train only)
    "emb_noise_std": 0.05,      # gaussian noise on embeddings (train only)
    "label_smoothing": 0.05,
    "mil_epochs": 30,
    "mil_lr": 2e-4,
    "mil_weight_decay": 2e-4,
    "warmup_epochs": 2,
    "grad_clip": 5.0,
    "early_patience": 8,
    "num_workers": 4,
    "K_TRAIN": 1024,            # random patch subsample per bag per epoch (bag-level aug)
    "k_sample_inst": 8,         # CLAM instance-loss top/bottom-k
    "lam_inst": 0.3,
    "overfit_gap_thresh": 0.15,  # warn if (val_loss - train_loss) exceeds this
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE2ID = {"negative": 0, "itc": 1, "micro": 2, "macro": 3}
ID2STAGE = {v: k for k, v in STAGE2ID.items()}
N_CLASSES = len(STAGE2ID)

np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])


# ============================================================
# STAGE 1 — UNI2-h ENCODER (reads JPEG patches from tiled .h5 -> per-slide .pt embeddings)
# ============================================================
_MODEL: Optional[nn.Module] = None
_TRANSFORM = None


def load_uni2h(cache_dir: Path) -> tuple[nn.Module, "callable"]:
    """Loads UNI2-h once and caches it in module globals."""
    global _MODEL, _TRANSFORM
    if _MODEL is not None:
        return _MODEL, _TRANSFORM

    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/UNI2-h", filename="pytorch_model.bin",
        local_dir=str(cache_dir), token=hf_token, force_download=False,
    )
    model = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
        num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True, pretrained=False,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=True)

    transform = create_transform(**resolve_data_config({}, model=model))
    model = model.to(DEVICE).eval()
    _MODEL, _TRANSFORM = model, transform
    return model, transform


def decode_jpeg_patch(jpeg_bytes: np.ndarray) -> Image.Image:
    import io
    return Image.open(io.BytesIO(jpeg_bytes.tobytes())).convert("RGB")


def load_h5_patch_indices(h5_path: Path, k_max: int, rng: np.random.RandomState) -> np.ndarray:
    """Tissue filtering already happened at tiling time (tissue_thresh in TilingConfig),
    so here we only randomly subsample if there are more patches than k_max — no rescoring.
    h5py fancy-indexing requires strictly increasing indices, hence the sort."""
    with h5py.File(h5_path, "r") as f:
        n = len(f["coords"])
    if n <= k_max:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k_max, replace=False))


def encode_patches_from_h5(h5_path: Path, model: nn.Module, transform, indices: np.ndarray,
                            batch_size: int, num_workers: int, prefetch: int) -> tuple[np.ndarray, np.ndarray]:
    """Decodes JPEG patches + coords for `indices` from an .h5 tile file and runs them
    through the UNI2-h encoder. Returns (features, coords)."""
    if len(indices) == 0:
        return np.array([]).reshape(0, 1536), np.array([]).reshape(0, 2)

    with h5py.File(h5_path, "r") as f:
        raw_patches = [f["patches"][i] for i in indices]
        coords = f["coords"][indices]

    q: "queue.Queue" = queue.Queue(maxsize=prefetch)

    def producer() -> None:
        for i in range(0, len(raw_patches), batch_size):
            batch_raw = raw_patches[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                tensors = [transform(img) for img in ex.map(decode_jpeg_patch, batch_raw)]
            if tensors:
                q.put(torch.stack(tensors))
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    features = []
    with torch.no_grad():
        while True:
            batch = q.get()
            if batch is None:
                break
            emb = model(batch.to(DEVICE, non_blocking=True))
            features.append(emb.cpu().numpy())
    t.join()
    return (np.vstack(features) if features else np.array([]).reshape(0, 1536)), coords


def encode_h5_directory(h5_dir: Path, output_dir: Path, cfg: dict, gpu: int = 0,
                         split: int = 0, total_splits: int = 1) -> None:
    """Encodes every tiled *.h5 under h5_dir into <slide_stem>.pt feature files."""
    if not h5_dir.exists():
        print(f"ERROR: {h5_dir} not found!")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    model, transform = load_uni2h(Path(cfg["cache_dir"]))
    rng = np.random.RandomState(cfg["seed"])

    all_h5 = sorted(h5_dir.rglob("*.h5"))
    h5_files = all_h5[split::total_splits]
    print(f"[encode] {h5_dir} -> {output_dir} | {len(h5_files)}/{len(all_h5)} files (split {split}/{total_splits})")

    processed = skipped = errors = 0
    with tqdm(total=len(h5_files), desc=f"encode gpu{gpu}") as pbar:
        for h5_path in h5_files:
            slide_name = h5_path.stem
            output_file = output_dir / f"{slide_name}.pt"
            if output_file.exists():
                skipped += 1
                pbar.update(1)
                continue
            try:
                indices = load_h5_patch_indices(h5_path, cfg["k_max_patches"], rng)
                if len(indices) == 0:
                    pbar.write(f"skip {slide_name} — empty .h5")
                    pbar.update(1)
                    continue
                features, coords = encode_patches_from_h5(
                    h5_path, model, transform, indices,
                    cfg["encode_batch_size"], cfg["encode_num_workers"], cfg["encode_prefetch"],
                )
                if features.shape[0] == 0:
                    pbar.write(f"skip {slide_name} — empty features")
                    pbar.update(1)
                    continue
                parts = slide_name.split("_")
                torch.save({
                    "features": features, "coords": coords, "slide_name": slide_name,
                    "patient_id": int(parts[1]), "node_id": int(parts[3]), "num_patches": len(indices),
                }, output_file)
                processed += 1
            except Exception as e:
                pbar.write(f"ERROR {slide_name}: {str(e)[:120]}")
                errors += 1
            pbar.update(1)
    print(f"[encode] done: processed={processed} skipped={skipped} errors={errors}")


# ============================================================
# STAGE 2 — DATASET + AUGMENTATION
# ============================================================
def augment_embeddings(emb: torch.Tensor, feat_dropout_p: float, noise_std: float, training: bool) -> torch.Tensor:
    """Embedding-space augmentation, train-only:
    1) feature dropout — zero random embedding dims (same mask across all instances in the bag)
    2) gaussian jitter — perturbs each instance independently, acts like patch-level noise
    """
    if not training:
        return emb
    if feat_dropout_p > 0:
        mask = torch.bernoulli(torch.full((emb.shape[1],), 1.0 - feat_dropout_p, device=emb.device))
        emb = emb * mask.unsqueeze(0)
    if noise_std > 0:
        emb = emb + torch.randn_like(emb) * noise_std
    return emb


class SlideEmbDataset(Dataset):
    """K is the random-subsample cap; resampled every __getitem__ call, so it also acts
    as a bag-level augmentation (different instance subset each epoch)."""

    def __init__(self, df: pd.DataFrame, uid2path: dict, K: Optional[int] = None):
        self.df, self.uid2path, self.K = df.reset_index(drop=True), uid2path, K

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        data = torch.load(self.uid2path[row["slide_uid"]], map_location="cpu", weights_only=False)
        emb = torch.tensor(data["features"], dtype=torch.float32)
        if self.K is not None and emb.shape[0] > self.K:
            emb = emb[torch.randperm(emb.shape[0])[: self.K]]
        return emb, int(row["stage_id"]), row["slide_uid"]


def collate_fn(batch):
    embs, stage_ids, uids = zip(*batch)
    return list(embs), list(stage_ids), list(uids)


# ============================================================
# STAGE 3 — CLAM-SB MODEL (gated attention + instance clustering loss)
# ============================================================
class GatedAttention(nn.Module):
    def __init__(self, hid: int, attn_dim: int):
        super().__init__()
        self.V = nn.Linear(hid, attn_dim)
        self.U = nn.Linear(hid, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)


class CLAM_SB(nn.Module):
    """Single-branch CLAM: shared gated attention + per-class binary instance
    classifiers used for the instance-clustering auxiliary loss."""

    def __init__(self, d: int, hid: int, n_classes: int, dropout: float, attn_dim: int):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d, hid), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention = GatedAttention(hid, attn_dim)
        self.bag_classifier = nn.Linear(hid, n_classes)
        self.inst_classifiers = nn.ModuleList([nn.Linear(hid, 1) for _ in range(n_classes)])
        self.n_classes = n_classes

    def forward(self, emb: torch.Tensor):
        h = self.feat(emb)                       # (M, hid)
        attn_w = torch.softmax(self.attention(h), dim=0)  # (M,)
        z = (attn_w.unsqueeze(1) * h).sum(0)      # (hid,)
        bag_logits = self.bag_classifier(z).unsqueeze(0)  # (1, n_classes)
        return bag_logits, attn_w, h


def clam_instance_loss(h: torch.Tensor, attn_w: torch.Tensor, bag_label: int,
                        inst_classifiers: nn.ModuleList, n_classes: int, k_sample: int) -> torch.Tensor:
    """CLAM instance-clustering loss: for the true class, top-attended patches are
    positive and bottom-attended are negative instances; for other classes, top-attended
    patches are treated as negative evidence."""
    M = h.shape[0]
    k = min(k_sample, M)
    top_idx = torch.topk(attn_w, k, largest=True).indices
    bot_idx = torch.topk(attn_w, k, largest=False).indices

    total = h.new_zeros(())
    for c in range(n_classes):
        clf = inst_classifiers[c]
        if c == bag_label:
            pos_logits = clf(h[top_idx]).squeeze(-1)
            neg_logits = clf(h[bot_idx]).squeeze(-1)
            logits = torch.cat([pos_logits, neg_logits])
            targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        else:
            logits = clf(h[top_idx]).squeeze(-1)
            targets = torch.zeros_like(logits)
        total = total + F.binary_cross_entropy_with_logits(logits, targets)
    return total / n_classes


def lr_cosine(epoch: int, base_lr: float, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    if warmup > 0 and epoch <= warmup:
        return base_lr * epoch / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return float(base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))))


# ============================================================
# STAGE 4 — TRAIN / EVAL
# ============================================================
@torch.no_grad()
def evaluate(model: CLAM_SB, loader: DataLoader, criterion: nn.Module, desc: Optional[str] = None) -> dict:
    model.eval()
    y_true, y_pred, y_prob_tumor, total_loss, n = [], [], [], 0.0, 0
    for embs, stage_ids, _ in tqdm(loader, total=len(loader), desc=desc, leave=False, disable=desc is None):
        for emb, sid in zip(embs, stage_ids):
            emb = emb.to(DEVICE)
            y_t = torch.tensor([sid], device=DEVICE)
            bag_logits, _, _ = model(emb)
            total_loss += float(criterion(bag_logits, y_t).item())
            probs = torch.softmax(bag_logits, dim=1).squeeze(0)
            y_true.append(sid)
            y_pred.append(int(probs.argmax().item()))
            y_prob_tumor.append(float(1.0 - probs[0].item()))  # P(not negative)
            n += 1
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic") if len(set(y_true)) > 1 else float("nan")
    y_bin = (y_true > 0).astype(int)
    auc = roc_auc_score(y_bin, y_prob_tumor) if len(set(y_bin)) > 1 else float("nan")
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)
    # per-class F1: без averaging, щоб бачити itc/micro окремо — macroF1 сам по собі
    # ховає, чи низький середній показник тягне один клас, чи всі помірно слабкі
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    return {
        "loss": total_loss / max(1, n),
        "qwk": qwk,
        "auc": auc,
        "macro_f1": macro_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "per_class_f1": {ID2STAGE[i]: float(per_class_f1[i]) for i in range(N_CLASSES)},
    }


def train_clam(train_df: pd.DataFrame, val_df: pd.DataFrame, uid2path: dict, cfg: dict) -> GraphCLAM_SB:
    print("\n" + "=" * 55 + "\nGraph-CLAM training (GATv2 context + gated attention + inst loss)\n" + "=" * 55)
    wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg, reinit=True)

    train_ds = SlideEmbDataset(train_df, uid2path, K=cfg["K_TRAIN"])
    val_ds = SlideEmbDataset(val_df, uid2path, K=None)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=cfg["num_workers"], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_fn)

    counts = train_df["stage_id"].value_counts().sort_index()
    class_w = torch.tensor([1.0 / max(1, counts.get(i, 1)) for i in range(N_CLASSES)], dtype=torch.float32, device=DEVICE)
    class_w /= class_w.sum()
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg["label_smoothing"])

    model = CLAM_SB(d=cfg["d_emb"], hid=cfg["hid"], n_classes=N_CLASSES, dropout=cfg["dropout"], attn_dim=cfg["attn_dim"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["mil_lr"], weight_decay=cfg["mil_weight_decay"])

    best_score, best_qwk_at_best, best_f1_at_best, best_state, patience = -1.0, None, None, None, 0

    for epoch in range(1, cfg["mil_epochs"] + 1):
        lr = lr_cosine(epoch, cfg["mil_lr"], cfg["mil_epochs"], cfg["warmup_epochs"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        train_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Ep {epoch:02d} train", leave=False)
        for embs, stage_ids, _ in pbar:
            for emb, sid in zip(embs, stage_ids):
                emb = augment_embeddings(emb.to(DEVICE), cfg["feat_dropout"], cfg["emb_noise_std"], training=True)
                y_t = torch.tensor([sid], device=DEVICE)

                optimizer.zero_grad()
                bag_logits, attn_w, h = model(emb)
                bag_loss = criterion(bag_logits, y_t)
                inst_loss = clam_instance_loss(h, attn_w, sid, model.inst_classifiers, N_CLASSES, cfg["k_sample_inst"])
                loss = bag_loss + cfg["lam_inst"] * inst_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                train_loss += float(loss.item())
                seen += 1
            pbar.set_postfix(loss=f"{train_loss / max(1, seen):.4f}")

        train_loss /= max(1, len(train_ds))
        val_metrics = evaluate(model, val_loader, criterion, desc=f"Ep {epoch:02d} val")
        val_loss, val_qwk, val_auc = val_metrics["loss"], val_metrics["qwk"], val_metrics["auc"]
        # Selection score: QWK alone is noisy on a small val set and tolerant of
        # confusion between adjacent classes (itc/micro/macro) — blending in macroF1
        # forces selection to also care about the rare classes.
        val_score = 0.5 * val_qwk + 0.5 * val_metrics["macro_f1"]
        gap = val_loss - train_loss
        overfit_flag = " ⚠ overfitting gap" if gap > cfg["overfit_gap_thresh"] else ""

        print(f"Ep {epoch:02d} | lr={lr:.2e} | train={train_loss:.4f} | val={val_loss:.4f} "
              f"| qwk={val_qwk:.4f} | auc={val_auc:.4f} | macroF1={val_metrics['macro_f1']:.4f} "
              f"| prec={val_metrics['macro_prec']:.4f} | rec={val_metrics['macro_rec']:.4f} "
              f"| score={val_score:.4f}{overfit_flag}"
              f"{' ⭐' if val_score > best_score else ''}")
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/qwk": val_qwk,
            "val/auc": val_auc,
            "val/macro_f1": val_metrics["macro_f1"],
            "val/macro_prec": val_metrics["macro_prec"],
            "val/macro_rec": val_metrics["macro_rec"],
            "val/score": val_score,
            "train_val_gap": gap,
        })

        if val_score > best_score:
            best_score, best_qwk_at_best, best_f1_at_best = val_score, val_qwk, val_metrics["macro_f1"]
            best_state, patience = {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            os.makedirs(cfg["save_dir"], exist_ok=True)
            torch.save(best_state, os.path.join(cfg["save_dir"], "best_clam.pt"))
        else:
            patience += 1
            if patience >= cfg["early_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\n✅ best val score={best_score:.4f} (qwk={best_qwk_at_best:.4f}, macroF1={best_f1_at_best:.4f})")
    wandb.finish()
    return model


def build_merged_stratified_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Merges both labeled pools (patient_000-099 via labels_csv + patient_100-199
    via xlsx) into one dataframe, then runs a *patient-level* stratified
    train/val/test split over the merged whole. Patient-level (not slide-level) so
    all nodes of one patient land in the same split — otherwise node_0 in train and
    node_1 in val could leak patient-specific tissue/staining signal. Stratified by
    each patient's worst (max) node stage, since that's the clinically meaningful
    summary and keeps itc/micro/macro balance across splits despite their rarity."""
    val_size, test_size = cfg["val_size"], cfg["test_size"]

    # pool A: patient_000-099 (labels_csv)
    train_feat_dir = Path(cfg["train_features_root"])
    uid2path = {p.stem: str(p) for p in sorted(train_feat_dir.glob("*.pt"))}
    df_raw = pd.read_csv(cfg["labels_csv"])
    df_a = df_raw[df_raw["patient"].str.contains(r"node_\d+", na=False)].copy()
    df_a["slide_uid"] = df_a["patient"].str.replace(r"\.tif$", "", regex=True).str.strip()
    df_a["stage"] = df_a["stage"].astype(str).str.lower().str.strip()
    df_a = df_a[df_a["stage"].isin(STAGE2ID)][["slide_uid", "stage"]]

    # pool B: patient_100-199 (xlsx)
    test_feat_dir = Path(cfg["test_features_dir"])
    uid2path_b = {p.stem: str(p) for p in sorted(test_feat_dir.glob("*.pt"))}
    df_b_raw = pd.read_excel(cfg["test_labels_xlsx"], sheet_name=0)
    df_b = df_b_raw[df_b_raw["slide"].str.match(r"^patient_\d+_node_\d+$", na=False)].copy()
    df_b["slide_uid"] = df_b["slide"].str.strip()
    df_b["stage"] = df_b["label"].astype(str).str.lower().str.strip()
    df_b = df_b[df_b["stage"].isin(STAGE2ID)]
    pid_b = df_b["slide_uid"].str.extract(r"patient_(\d+)_node_\d+")[0].astype(int)
    df_b = df_b[(pid_b >= cfg["test_patient_lo"]) & (pid_b <= cfg["test_patient_hi"])][["slide_uid", "stage"]]

    # merge (patient id ranges 000-099 vs 100-199 don't collide)
    df = pd.concat([df_a, df_b], ignore_index=True)
    uid2path.update(uid2path_b)
    df["stage_id"] = df["stage"].map(STAGE2ID).astype(int)
    df["patient_id"] = df["slide_uid"].str.extract(r"patient_(\d+)")[0].astype(int)
    df = df[df["slide_uid"].isin(set(uid2path.keys()))].reset_index(drop=True)

    # patient-level stratified split, key = worst (max) node stage per patient
    patient_key = df.groupby("patient_id")["stage_id"].max()
    patients, strata = patient_key.index.to_numpy(), patient_key.to_numpy()

    train_pts, holdout_pts = train_test_split(
        patients, test_size=val_size + test_size, stratify=strata, random_state=cfg["seed"])
    holdout_strata = patient_key.loc[holdout_pts].to_numpy()
    val_pts, test_pts = train_test_split(
        holdout_pts, test_size=test_size / (val_size + test_size),
        stratify=holdout_strata, random_state=cfg["seed"])

    train_df = df[df["patient_id"].isin(train_pts)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pts)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pts)].reset_index(drop=True)

    print(f"merged pool: {len(df)} слайдів, {len(patients)} пацієнтів")
    print(f"train: {len(train_df)} слайдів ({len(train_pts)} пац.) | "
          f"val: {len(val_df)} слайдів ({len(val_pts)} пац.) | "
          f"test: {len(test_df)} слайдів ({len(test_pts)} пац.)")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d["stage"].value_counts().reindex(STAGE2ID.keys(), fill_value=0)
        print(f"  {name}: {counts.to_dict()}")

    return train_df, val_df, test_df, uid2path


def eval_labeled_set(name: str, df: pd.DataFrame, uid2path: dict, model: CLAM_SB, cfg: dict) -> dict:
    """Runs `evaluate()` over any labeled slide set with the already-trained
    model and prints the same metrics used during training."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    ds = SlideEmbDataset(df, uid2path, K=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_fn)
    metrics = evaluate(model, loader, criterion)
    print(f"\n[{name}] n={len(df)} | loss={metrics['loss']:.4f} | qwk={metrics['qwk']:.4f} "
          f"| auc={metrics['auc']:.4f} | macroF1={metrics['macro_f1']:.4f} "
          f"| prec={metrics['macro_prec']:.4f} | rec={metrics['macro_rec']:.4f}")
    pcf1 = metrics["per_class_f1"]
    print(f"  per-class F1: negative={pcf1['negative']:.4f} | itc={pcf1['itc']:.4f} "
          f"| micro={pcf1['micro']:.4f} | macro={pcf1['macro']:.4f}")
    return metrics


# ============================================================
# STAGE 5 — INFERENCE / SUBMISSION
# ============================================================
def nodes_to_pN(node_stages: list[str]) -> str:
    n_macro, n_micro, n_itc = node_stages.count("macro"), node_stages.count("micro"), node_stages.count("itc")
    if n_macro >= 4:
        return "pN3"
    if n_macro >= 2:
        return "pN2"
    if n_macro == 1:
        return "pN1"
    if n_micro >= 1:
        return "pN1mi"
    if n_itc >= 1:
        return "pN0(i+)"
    return "pN0"


@torch.no_grad()
def run_inference(cfg: dict) -> None:
    model = CLAM_SB(d=cfg["d_emb"], hid=cfg["hid"], n_classes=N_CLASSES, dropout=cfg["dropout"], attn_dim=cfg["attn_dim"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(os.path.join(cfg["save_dir"], "best_clam.pt"), map_location=DEVICE, weights_only=False))

    pt_files = sorted(Path(cfg["test_features_dir"]).glob("*.pt"))
    print(f"Тестових .pt файлів: {len(pt_files)}")

    slide_preds: dict[str, str] = {}
    for pt_path in tqdm(pt_files, desc="Inference"):
        slide_name = pt_path.stem
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        emb = torch.tensor(data["features"], dtype=torch.float32).to(DEVICE)
        bag_logits, _, _ = model(emb)
        pred = int(bag_logits.argmax(dim=1).item())
        slide_preds[slide_name] = ID2STAGE[pred]

    patient_nodes: dict[int, dict[int, str]] = {}
    for slide_name, stage in slide_preds.items():
        parts = slide_name.split("_")
        patient_nodes.setdefault(int(parts[1]), {})[int(parts[3])] = stage

    rows = []
    for patient_id in sorted(patient_nodes):
        nodes = patient_nodes[patient_id]
        node_list = [nodes.get(i, "negative") for i in range(5)]
        patient_str = f"patient_{patient_id:03d}"
        rows.append({"patient": f"{patient_str}.zip", "stage": nodes_to_pN(node_list)})
        for i, stage in enumerate(node_list):
            rows.append({"patient": f"{patient_str}_node_{i}.tif", "stage": stage})

    df_submission = pd.DataFrame(rows)
    out_path = Path(cfg["submission_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(out_path, index=False)

    print(df_submission.head(12).to_string(index=False))
    print(f"\npN distribution:\n{df_submission[df_submission['patient'].str.endswith('.zip')]['stage'].value_counts()}")
    print(f"\n✅ Submission збережено: {out_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-train", action="store_true")
    parser.add_argument("--encode-test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--eval-train", action="store_true", help="metrics on the merged-split train/val sets")
    parser.add_argument("--eval-test", action="store_true", help="metrics on the merged-split held-out test set")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--total-splits", type=int, default=1)
    args = parser.parse_args()

    if args.encode_train:
        encode_h5_directory(Path(CFG["train_h5_dir"]), Path(CFG["train_features_root"]),
                             CFG, gpu=args.gpu, split=args.split, total_splits=args.total_splits)
    if args.encode_test:
        encode_h5_directory(Path(CFG["test_h5_dir"]), Path(CFG["test_features_dir"]),
                             CFG, gpu=args.gpu, split=args.split, total_splits=args.total_splits)
    if args.train:
        train_df, val_df, test_df, uid2path = build_merged_stratified_splits(CFG)
        train_clam(train_df, val_df, uid2path, CFG)
    if args.infer:
        run_inference(CFG)
    if args.eval_train or args.eval_test:
        eval_model = CLAM_SB(d=CFG["d_emb"], hid=CFG["hid"], n_classes=N_CLASSES, dropout=CFG["dropout"],
                              attn_dim=CFG["attn_dim"]).to(DEVICE).eval()
        eval_model.load_state_dict(torch.load(os.path.join(CFG["save_dir"], "best_clam.pt"),
                                               map_location=DEVICE, weights_only=False))
        train_df, val_df, test_df, uid2path = build_merged_stratified_splits(CFG)
        if args.eval_train:
            eval_labeled_set("TRAIN split", train_df, uid2path, eval_model, CFG)
            eval_labeled_set("VAL split (checkpoint selection)", val_df, uid2path, eval_model, CFG)
        if args.eval_test:
            eval_labeled_set("TEST split (held out)", test_df, uid2path, eval_model, CFG)
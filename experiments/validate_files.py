"""Validate tiled .h5 files: patch_level consistency, coords/patches integrity, edge-coverage bug impact."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import openslide


def check_h5(h5_path: Path, slide_dir: Path, expected_level: int) -> dict:
    result = {"name": h5_path.stem, "ok": True, "issues": []}

    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" not in f or "patches" not in f:
                result["issues"].append("missing coords/patches dataset")
                result["ok"] = False
                return result

            n_coords = len(f["coords"])
            n_patches = len(f["patches"])
            level = f.attrs.get("patch_level", None)

            result["n_patches"] = n_coords

            if n_coords != n_patches:
                result["issues"].append(f"length mismatch: coords={n_coords} patches={n_patches}")
                result["ok"] = False

            if n_coords == 0:
                result["issues"].append("empty (0 patches) — check tissue mask / sat_thresh")
                result["ok"] = False

            if level != expected_level:
                result["issues"].append(f"patch_level={level}, expected {expected_level}")
                result["ok"] = False

            # spot-check a few patches actually decode (catch truncated JPEG from crashed writes)
            if n_patches > 0:
                check_idx = [0, n_patches // 2, n_patches - 1]
                for i in check_idx:
                    raw = f["patches"][i]
                    if raw.nbytes == 0:
                        result["issues"].append(f"patch {i} is empty buffer (likely truncated write)")
                        result["ok"] = False

    except Exception as e:
        result["issues"].append(f"failed to open/read: {e}")
        result["ok"] = False
        return result

    # edge-coverage diagnostic: how much tissue area is lost to the range() truncation bug
    tif_path = slide_dir / f"{h5_path.stem}.tif"
    if tif_path.exists():
        try:
            edge_loss_pct = estimate_edge_loss(tif_path, expected_level)
            result["edge_loss_pct"] = edge_loss_pct
        except Exception as e:
            result["issues"].append(f"edge-loss check failed: {e}")

    return result


def estimate_edge_loss(tif_path: Path, patch_level: int, mask_level_offset: int = 3, patch_size: int = 256) -> float:
    """Fraction of mask height/width that falls in the truncated last strip (range(0, n-step, step) bug)."""
    import cv2

    slide = openslide.OpenSlide(str(tif_path))
    mask_level = min(patch_level + mask_level_offset, slide.level_count - 1)
    thumb = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert("RGB")
    hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask > 0

    downsample = slide.level_downsamples[patch_level] / slide.level_downsamples[mask_level]
    step = round(patch_size * downsample)
    mask_h, mask_w = mask.shape

    last_row_start = ((mask_h - step) // step) * step if mask_h > step else 0
    last_col_start = ((mask_w - step) // step) * step if mask_w > step else 0

    lost_area = mask[last_row_start:, :].sum() + mask[:, last_col_start:].sum() - mask[last_row_start:, last_col_start:].sum()
    total_tissue = mask.sum()
    return 100.0 * lost_area / total_tissue if total_tissue > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--slide-dir", type=Path, required=True, help="dir with original .tif (for edge-loss check)")
    parser.add_argument("--expected-level", type=int, default=1)
    parser.add_argument("--skip-edge-check", action="store_true", help="skip slow edge-loss diagnostic")
    args = parser.parse_args()

    h5_files = sorted(args.h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"no .h5 files found in {args.h5_dir}")
        return

    print(f"checking {len(h5_files)} files...\n")

    bad = []
    edge_losses = []
    for h5_path in h5_files:
        r = check_h5(h5_path, args.slide_dir if not args.skip_edge_check else Path("/nonexistent"), args.expected_level)
        status = "OK" if r["ok"] else "FAIL"
        extra = f", n_patches={r.get('n_patches', '?')}"
        if "edge_loss_pct" in r:
            extra += f", edge_loss={r['edge_loss_pct']:.2f}%"
            edge_losses.append(r["edge_loss_pct"])
        print(f"[{status}] {r['name']}{extra}")
        for issue in r["issues"]:
            print(f"    -> {issue}")
        if not r["ok"]:
            bad.append(r["name"])

    print(f"\n{'='*50}")
    print(f"{len(h5_files) - len(bad)}/{len(h5_files)} passed")
    if bad:
        print(f"FAILED ({len(bad)}): {bad}")
        print("\ndelete and re-tile these:")
        for name in bad:
            print(f"  rm {args.h5_dir / (name + '.h5')}")
    if edge_losses:
        print(f"\nedge-loss stats: mean={np.mean(edge_losses):.2f}%, max={np.max(edge_losses):.2f}%")


if __name__ == "__main__":
    main()
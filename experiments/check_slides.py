"""Benchmark: full slide -> tissue patches at level=1, compare raw vs JPEG-in-h5 storage."""
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image


@dataclass
class SlideStats:
    name: str
    tif_size_mb: float
    n_grid_patches: int = 0
    n_tissue_patches: int = 0
    raw_h5_size_mb: float = 0.0
    jpeg_h5_size_mb: float = 0.0
    extract_time_s: float = 0.0


def build_tissue_mask(slide: openslide.OpenSlide, mask_level: int, sat_thresh: int = 20) -> np.ndarray:
    """Otsu on saturation channel; returns binary mask at mask_level resolution."""
    thumb = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert("RGB")
    hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, sat_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def iter_tissue_patch_coords(
    slide: openslide.OpenSlide,
    mask: np.ndarray,
    mask_level: int,
    patch_level: int,
    patch_size: int,
    tissue_thresh: float = 0.5,
) -> tuple[list[tuple[int, int]], int]:
    """Returns (tissue coords at level0 pixel space, total grid count)."""
    downsample_patch_to_mask = (
        slide.level_downsamples[patch_level] / slide.level_downsamples[mask_level]
    )
    step_in_mask = round(patch_size * downsample_patch_to_mask)  # округлення, не int()-truncation
    scale_mask_to_level0 = slide.level_downsamples[mask_level]

    coords: list[tuple[int, int]] = []
    n_grid = 0
    mask_h, mask_w = mask.shape
    for my in range(0, mask_h - step_in_mask, step_in_mask):
        for mx in range(0, mask_w - step_in_mask, step_in_mask):
            n_grid += 1
            tile = mask[my : my + step_in_mask, mx : mx + step_in_mask]
            if tile.mean() > tissue_thresh:
                coords.append((int(mx * scale_mask_to_level0), int(my * scale_mask_to_level0)))
    return coords, n_grid


def extract_and_store(
    slide_path: Path,
    coords: list[tuple[int, int]],
    patch_level: int,
    patch_size: int,
    raw_out: Path,
    jpeg_out: Path,
    jpeg_quality: int = 90,
) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    n = len(coords)

    with h5py.File(raw_out, "w") as f_raw, h5py.File(jpeg_out, "w") as f_jpg:
        raw_ds = f_raw.create_dataset(
            "patches", shape=(n, patch_size, patch_size, 3), dtype=np.uint8, compression="lzf"
        )
        f_raw.create_dataset("coords", data=np.array(coords, dtype=np.int32))

        jpg_ds = f_jpg.create_dataset("patches", shape=(n,), dtype=h5py.vlen_dtype(np.uint8))
        f_jpg.create_dataset("coords", data=np.array(coords, dtype=np.int32))

        for i, coord in enumerate(coords):
            patch = slide.read_region(coord, patch_level, (patch_size, patch_size)).convert("RGB")
            arr = np.array(patch)
            raw_ds[i] = arr

            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=jpeg_quality)
            jpg_ds[i] = np.frombuffer(buf.getvalue(), dtype=np.uint8)


def benchmark_slide(
    slide_path: Path,
    out_dir: Path,
    patch_level: int = 1,
    patch_size: int = 256,
    mask_level_offset: int = 3,
) -> SlideStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = SlideStats(name=slide_path.stem, tif_size_mb=slide_path.stat().st_size / 1e6)

    slide = openslide.OpenSlide(str(slide_path))
    mask_level = min(patch_level + mask_level_offset, slide.level_count - 1)
    mask = build_tissue_mask(slide, mask_level)

    coords, n_grid = iter_tissue_patch_coords(slide, mask, mask_level, patch_level, patch_size)
    stats.n_grid_patches = n_grid
    stats.n_tissue_patches = len(coords)

    raw_out = out_dir / f"{slide_path.stem}_raw.h5"
    jpeg_out = out_dir / f"{slide_path.stem}_jpeg.h5"

    t0 = time.perf_counter()
    extract_and_store(slide_path, coords, patch_level, patch_size, raw_out, jpeg_out)
    stats.extract_time_s = time.perf_counter() - t0

    stats.raw_h5_size_mb = raw_out.stat().st_size / 1e6
    stats.jpeg_h5_size_mb = jpeg_out.stat().st_size / 1e6
    return stats


def print_report(all_stats: list[SlideStats]) -> None:
    header = (
        f"{'slide':<20}{'tif_MB':>10}{'grid_n':>10}{'tissue_n':>10}"
        f"{'tissue_%':>10}{'raw_MB':>10}{'jpeg_MB':>10}{'jpeg/tif_%':>12}{'time_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in all_stats:
        tissue_pct = 100 * s.n_tissue_patches / max(s.n_grid_patches, 1)
        jpeg_ratio = 100 * s.jpeg_h5_size_mb / max(s.tif_size_mb, 1e-9)
        print(
            f"{s.name:<20}{s.tif_size_mb:>10.1f}{s.n_grid_patches:>10}{s.n_tissue_patches:>10}"
            f"{tissue_pct:>9.1f}%{s.raw_h5_size_mb:>10.1f}{s.jpeg_h5_size_mb:>10.1f}"
            f"{jpeg_ratio:>11.1f}%{s.extract_time_s:>8.1f}"
        )


if __name__ == "__main__":
    SLIDE_PATHS = [
        Path("/data/bohatyrenko1/CAMELYON_test/patient_100_node_0.tif")
    ]
    OUT_DIR = Path("/data/bohatyrenko1/patch_bench")

    results = [benchmark_slide(p, OUT_DIR) for p in SLIDE_PATHS if p.exists()]
    print_report(results)
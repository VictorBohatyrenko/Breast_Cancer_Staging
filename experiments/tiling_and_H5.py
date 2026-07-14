"""Tile CAMELYON17 WSIs (patients 100-119) into tissue patches, JPEG-encoded in HDF5."""
import io
import logging
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] %(message)s")
logger = logging.getLogger(__name__)

SLIDE_RE = re.compile(r"patient_(\d+)_node_(\d+)\.tif$")


@dataclass(frozen=True)
class TilingConfig:
    patch_level: int = 1  # 20x, matches UNI2-h training resolution
    patch_size: int = 256
    mask_level_offset: int = 3  # mask_level = patch_level + offset, clamped to level_count-1
    tissue_thresh: float = 0.15
    sat_thresh: int = 20  # Otsu seed threshold on HSV saturation channel
    jpeg_quality: int = 90
    n_workers: int = 8


def pick_mask_level(slide: openslide.OpenSlide, patch_level: int, offset: int) -> int:
    """Clamp mask_level to valid range so tiny pyramids don't overflow level_count."""
    return min(patch_level + offset, slide.level_count - 1)


def build_tissue_mask(slide: openslide.OpenSlide, mask_level: int, sat_thresh: int) -> np.ndarray:
    thumb = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert("RGB")
    hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], sat_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def iter_tissue_patch_coords(
    slide: openslide.OpenSlide, mask: np.ndarray, mask_level: int, cfg: TilingConfig
) -> list[tuple[int, int]]:
    """Level0-pixel-space coords of patches whose tissue fraction exceeds cfg.tissue_thresh."""
    downsample = slide.level_downsamples[cfg.patch_level] / slide.level_downsamples[mask_level]
    step = round(cfg.patch_size * downsample)  # round(), not int()-truncation — fixed bug
    scale_to_level0 = slide.level_downsamples[mask_level]

    coords: list[tuple[int, int]] = []
    mask_h, mask_w = mask.shape
    for my in range(0, mask_h - step, step):
        for mx in range(0, mask_w - step, step):
            if mask[my : my + step, mx : mx + step].mean() > cfg.tissue_thresh:
                coords.append((int(mx * scale_to_level0), int(my * scale_to_level0)))
    return coords


def encode_jpeg(patch_rgb: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(patch_rgb).save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def tile_slide(slide_path: Path, out_dir: Path, cfg: TilingConfig) -> tuple[str, int]:
    """Tiles one slide to <stem>.h5 (JPEG-encoded patches + level0 coords). Returns (name, n_patches).

    A partially-written .h5 (e.g. from a prior 'No space left on device' crash) has no
    valid 'coords' dataset — we treat that as not-yet-tiled and overwrite it rather than
    silently skipping.
    """
    out_path = out_dir / f"{slide_path.stem}.h5"
    if out_path.exists():
        try:
            with h5py.File(out_path, "r") as f:
                n_coords = len(f["coords"])
            if n_coords > 0:
                logger.info("skip %s (already tiled, %d patches)", slide_path.stem, n_coords)
                return slide_path.stem, n_coords
            logger.warning("re-tiling %s (existing .h5 has empty coords)", slide_path.stem)
        except Exception:
            logger.warning("re-tiling %s (existing .h5 is corrupt/unreadable)", slide_path.stem)

    slide = openslide.OpenSlide(str(slide_path))
    mask_level = pick_mask_level(slide, cfg.patch_level, cfg.mask_level_offset)
    mask = build_tissue_mask(slide, mask_level, cfg.sat_thresh)
    coords = iter_tissue_patch_coords(slide, mask, mask_level, cfg)

    with h5py.File(out_path, "w") as f:
        patches_ds = f.create_dataset("patches", shape=(len(coords),), dtype=h5py.vlen_dtype(np.uint8))
        f.create_dataset("coords", data=np.array(coords, dtype=np.int32))
        f.attrs["patch_level"] = cfg.patch_level
        f.attrs["patch_size"] = cfg.patch_size
        f.attrs["jpeg_quality"] = cfg.jpeg_quality

        for i, coord in enumerate(coords):
            patch = slide.read_region(coord, cfg.patch_level, (cfg.patch_size, cfg.patch_size)).convert("RGB")
            patches_ds[i] = encode_jpeg(np.array(patch), cfg.jpeg_quality)

    logger.info("tiled %s -> %d patches", slide_path.stem, len(coords))
    return slide_path.stem, len(coords)


def _tile_worker(args: tuple[Path, Path, TilingConfig]) -> tuple[str, int]:
    slide_path, out_dir, cfg = args
    try:
        return tile_slide(slide_path, out_dir, cfg)
    except Exception:
        logger.exception("failed on %s", slide_path.stem)
        return slide_path.stem, -1


def collect_slides(
    data_dir: Path,
    patient_lo: int,
    patient_hi: int,
    start_node_at_lo: int = 0,
) -> list[Path]:
    """Slides for patient in [patient_lo, patient_hi]. For patient == patient_lo,
    only node >= start_node_at_lo is included (resume point). Walks recursively so it
    works for both flat dirs and center_*/patient_*/ nested layouts."""
    slides = []
    for p in sorted(data_dir.rglob("*.tif")):
        m = SLIDE_RE.search(p.name)
        if not m:
            continue
        patient, node = int(m.group(1)), int(m.group(2))
        if not (patient_lo <= patient <= patient_hi):
            continue
        if patient == patient_lo and node < start_node_at_lo:
            continue
        slides.append(p)
    return slides


def run(
    data_dir: Path,
    out_dir: Path,
    patient_lo: int,
    patient_hi: int,
    cfg: TilingConfig,
    start_node_at_lo: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = collect_slides(data_dir, patient_lo, patient_hi, start_node_at_lo)
    logger.info(
        "found %d slides for patients %d-%d (starting node %d at patient %d)",
        len(slides), patient_lo, patient_hi, start_node_at_lo, patient_lo,
    )
    if not slides:
        logger.warning("no slides matched — check data_dir / naming pattern")
        return

    with mp.Pool(cfg.n_workers) as pool:
        results = pool.map(_tile_worker, [(s, out_dir, cfg) for s in slides])

    ok = [r for r in results if r[1] >= 0]
    failed = [r for r in results if r[1] < 0]
    total_patches = sum(n for _, n in ok)
    logger.info("done: %d/%d slides ok, %d patches total", len(ok), len(slides), total_patches)
    if failed:
        logger.warning("failed slides: %s", [name for name, _ in failed])


if __name__ == "__main__":
    # patient_060/patient_060_node_0.tif lives here; rglob also picks up any siblings
    # (patient_061..073) if you point DATA_DIR at the center/training root instead.
    DATA_DIR = Path("/data/bohatyrenko1/CAMELYON/training/center_3")
    OUT_DIR = Path("/data/bohatyrenko1/patches_jpeg_level1_training")
    run(
        DATA_DIR,
        OUT_DIR,
        patient_lo=61,
        patient_hi=73,
        cfg=TilingConfig(),
        start_node_at_lo=0,
    )
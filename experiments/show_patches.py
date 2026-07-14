"""Visual QC: sample and render N patches from a tiled .h5 to inspect tissue/tiling quality."""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def decode_patch(raw: np.ndarray) -> np.ndarray:
    """raw is a vlen uint8 buffer (JPEG bytes) -> decoded HxWx3 RGB array."""
    return np.array(Image.open(io_bytes(raw)))


def io_bytes(raw: np.ndarray):
    import io
    return io.BytesIO(raw.tobytes())


def sample_patches(h5_path: Path, n: int, seed: int = 42) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    with h5py.File(h5_path, "r") as f:
        n_total = len(f["coords"])
        if n_total == 0:
            raise ValueError(f"{h5_path.name}: 0 patches — empty tissue mask, nothing to show")
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=min(n, n_total), replace=False)
        idx.sort()  # sequential read з h5py дешевше за random access по vlen dataset

        patches = [decode_patch(f["patches"][i]) for i in idx]
        coords = [tuple(f["coords"][i]) for i in idx]
    return patches, coords


def plot_grid(patches: list[np.ndarray], coords: list[tuple[int, int]], slide_name: str, out_path: Path, ncols: int = 5):
    n = len(patches)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = np.atleast_2d(axes).ravel()

    for ax, patch, (x, y) in zip(axes, patches, coords):
        ax.imshow(patch)
        ax.set_title(f"({x},{y})", fontsize=7)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{slide_name} — {n} random patches (level0 coords)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("/data/bohatyrenko1/patches_jpeg_level1"))
    parser.add_argument("--slide", type=str, required=True, help="stem, e.g. patient_185_node_3")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    h5_path = args.h5_dir / f"{args.slide}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    patches, coords = sample_patches(h5_path, args.n, args.seed)
    out_path = args.out or Path(f"/data/bohatyrenko1/qc_{args.slide}.png")
    plot_grid(patches, coords, args.slide, out_path)
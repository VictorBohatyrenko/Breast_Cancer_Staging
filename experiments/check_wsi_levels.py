"""
check_wsi_levels.py

Перевірка pyramid levels у WSI (TIFF/SVS/NDPI тощо) на предмет реального mpp/magnification
кожного рівня, з метою підібрати рівень, максимально близький до target magnification
foundation-моделі (за замовчуванням UNI2-h: 20x / 0.5 um/px).

Ключова проблема, яку цей скрипт вирішує:
    `level` в pyramidal TIFF -- це просто індекс даунсемплінгу, а НЕ magnification.
    Для generic TIFF (без vendor metadata) openslide.properties часто пустий --
    доводиться рахувати mpp вручну з TIFF ResolutionUnit/XResolution тегів.

Usage:
    python check_wsi_levels.py /path/to/slide.tif
    python check_wsi_levels.py /path/to/wsi_dir --target-mpp 0.5 --csv report.csv
    python check_wsi_levels.py /path/to/wsi_dir --recursive
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wsi_level_check")

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    log.warning("openslide-python не встановлено -- працюватиму тільки через tifffile fallback "
                "(pip install openslide-python, потрібна також системна бібліотека libopenslide).")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    log.warning("tifffile не встановлено -- fallback для generic TIFF без openslide metadata "
                "буде недоступний (pip install tifffile).")


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu"}

# ResolutionUnit TIFF tag: 1=None, 2=Inch, 3=Centimeter
_RESOLUTION_UNIT_TO_UM = {
    2: 25400.0,   # 1 inch = 25400 um
    3: 10000.0,   # 1 cm = 10000 um
}


@dataclasses.dataclass
class LevelInfo:
    level: int
    width: int
    height: int
    downsample: float
    mpp: Optional[float]  # None якщо неможливо визначити


@dataclasses.dataclass
class SlideReport:
    path: str
    reader: str  # "openslide" | "tifffile"
    base_mpp: Optional[float]
    base_mpp_source: str  # звідки взяли base_mpp: "openslide.mpp-x", "objective-power", "tiff-resolution-tag", "unknown"
    levels: list[LevelInfo]
    target_mpp: float
    best_level: Optional[int]
    best_level_mpp: Optional[float]
    warning: Optional[str]


def _mpp_from_tiff_tags(page) -> Optional[float]:
    """
    Рахує mpp з XResolution/ResolutionUnit тегів окремої TIFF-сторінки.
    Це fallback для generic TIFF без openslide vendor metadata.

    Обережно: багато сканерів/конвертерів пишуть ці теги некоректно або не пишуть взагалі --
    результат тут ЗАВЖДИ треба crosscheck-ати з тим, що реально відомо про сканування
    (протокол датасету, супровідна документація), а не довіряти сліпо.
    """
    tags = page.tags
    if "XResolution" not in tags or "ResolutionUnit" not in tags:
        return None

    x_res = tags["XResolution"].value  # зазвичай (numerator, denominator) або float
    res_unit = tags["ResolutionUnit"].value

    if isinstance(x_res, tuple):
        num, den = x_res
        if den == 0:
            return None
        x_res_val = num / den
    else:
        x_res_val = float(x_res)

    if x_res_val <= 0:
        return None

    unit_to_um = _RESOLUTION_UNIT_TO_UM.get(int(res_unit))
    if unit_to_um is None:
        # ResolutionUnit == 1 (none) -- пікселі/умовна одиниця, mpp невизначений
        return None

    # x_res_val = pixels per unit -> mpp = unit_in_um / pixels_per_unit
    mpp = unit_to_um / x_res_val
    return mpp


def _read_via_openslide(path: Path, target_mpp: float) -> Optional[SlideReport]:
    if not HAS_OPENSLIDE:
        return None
    try:
        slide = openslide.OpenSlide(str(path))
    except Exception as e:
        log.debug(f"openslide не зміг відкрити {path}: {e}")
        return None

    props = slide.properties
    base_mpp: Optional[float] = None
    source = "unknown"

    mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    objective_power = props.get("openslide.objective-power")

    if mpp_x is not None:
        try:
            base_mpp = float(mpp_x)
            source = "openslide.mpp-x"
        except ValueError:
            pass

    if base_mpp is None and objective_power is not None:
        try:
            # грубе наближення: 40x ~ 0.25 mpp, лінійна екстраполяція
            base_mpp = 40.0 / float(objective_power) * 0.25
            source = "objective-power (наближено)"
        except (ValueError, ZeroDivisionError):
            pass

    downsamples = list(slide.level_downsamples)
    dims = list(slide.level_dimensions)

    levels = []
    for lvl, (ds, (w, h)) in enumerate(zip(downsamples, dims)):
        lvl_mpp = base_mpp * ds if base_mpp is not None else None
        levels.append(LevelInfo(level=lvl, width=w, height=h, downsample=ds, mpp=lvl_mpp))

    slide.close()

    return _finalize_report(path, "openslide", base_mpp, source, levels, target_mpp)


def _read_via_tifffile(path: Path, target_mpp: float) -> Optional[SlideReport]:
    if not HAS_TIFFFILE:
        return None
    try:
        tf = tifffile.TiffFile(str(path))
    except Exception as e:
        log.debug(f"tifffile не зміг відкрити {path}: {e}")
        return None

    series = tf.series[0] if tf.series else None
    if series is None:
        return None

    # У pyramidal TIFF рівні часто представлені як series.levels (tifffile >= 2021.x)
    pages = series.levels if hasattr(series, "levels") and series.levels else [series]

    base_mpp = None
    source = "unknown"
    base_w = base_h = None

    levels: list[LevelInfo] = []
    for lvl, level_series in enumerate(pages):
        page = level_series.pages[0] if hasattr(level_series, "pages") else level_series[0]
        shape = level_series.shape if hasattr(level_series, "shape") else page.shape
        # shape зазвичай (H, W, C) або (H, W)
        h, w = shape[0], shape[1]

        mpp = _mpp_from_tiff_tags(page)
        if lvl == 0:
            base_mpp = mpp
            base_w, base_h = w, h
            source = "tiff-resolution-tag" if mpp is not None else "unknown"

        downsample = (base_w / w) if base_w else 1.0
        levels.append(LevelInfo(level=lvl, width=w, height=h, downsample=downsample, mpp=mpp))

    tf.close()
    return _finalize_report(path, "tifffile", base_mpp, source, levels, target_mpp)


def _finalize_report(
    path: Path,
    reader: str,
    base_mpp: Optional[float],
    source: str,
    levels: list[LevelInfo],
    target_mpp: float,
) -> SlideReport:
    best_level = None
    best_mpp = None
    warning = None

    known_mpp_levels = [lv for lv in levels if lv.mpp is not None]

    if not known_mpp_levels:
        warning = (
            "Не вдалося визначити mpp жодним способом (ні openslide metadata, ні TIFF resolution tags). "
            "Target magnification НЕ можна підтвердити автоматично -- потрібні зовнішні дані "
            "про протокол сканування (напр. з опису датасету)."
        )
    else:
        diffs = [(lv, abs(lv.mpp - target_mpp)) for lv in known_mpp_levels]
        best_level_info, best_diff = min(diffs, key=lambda t: t[1])
        best_level = best_level_info.level
        best_mpp = best_level_info.mpp

        rel_diff = best_diff / target_mpp
        if rel_diff > 0.15:
            warning = (
                f"Найближчий рівень (level={best_level}) має mpp={best_mpp:.3f}, "
                f"це відхиляється від target={target_mpp} на {rel_diff*100:.1f}%. "
                f"Жоден дискретний рівень пірамиди не відповідає точно 20x -- "
                f"розглянь custom downsampling через read_region з проміжним фактором, "
                f"а не прив'язку до дискретного level index."
            )

    return SlideReport(
        path=str(path),
        reader=reader,
        base_mpp=base_mpp,
        base_mpp_source=source,
        levels=levels,
        target_mpp=target_mpp,
        best_level=best_level,
        best_level_mpp=best_mpp,
        warning=warning,
    )


def analyze_slide(path: Path, target_mpp: float) -> SlideReport:
    """
    Пробує openslide спочатку (правильніше визначає vendor metadata),
    падає на tifffile fallback якщо openslide недоступний, не зміг відкрити,
    або не знайшов mpp (напр. generic TIFF без vendor tags -- тоді ручний
    розрахунок з resolution tags може спрацювати краще).
    """
    report = _read_via_openslide(path, target_mpp)

    if report is None or report.base_mpp is None:
        fallback_report = _read_via_tifffile(path, target_mpp)
        if fallback_report is not None and fallback_report.base_mpp is not None:
            if report is not None:
                log.info(f"{path.name}: openslide не дав mpp, використано tifffile fallback")
            report = fallback_report
        elif report is None:
            report = fallback_report

    if report is None:
        report = SlideReport(
            path=str(path), reader="none", base_mpp=None, base_mpp_source="unknown",
            levels=[], target_mpp=target_mpp, best_level=None, best_level_mpp=None,
            warning="Не вдалося відкрити файл жодним доступним читачем (openslide/tifffile).",
        )

    return report


def print_report(report: SlideReport) -> None:
    print(f"\n{'='*80}")
    print(f"Файл:        {report.path}")
    print(f"Reader:      {report.reader}")
    print(f"base_mpp:    {report.base_mpp if report.base_mpp else 'N/A'}  (джерело: {report.base_mpp_source})")
    print(f"target_mpp:  {report.target_mpp}  (20x еквівалент, для UNI2-h)")
    print(f"{'-'*80}")
    print(f"{'level':>5} | {'width':>7} | {'height':>7} | {'downsample':>10} | {'mpp':>8}")
    for lv in report.levels:
        mpp_str = f"{lv.mpp:.4f}" if lv.mpp is not None else "N/A"
        marker = "  <-- best" if lv.level == report.best_level else ""
        print(f"{lv.level:>5} | {lv.width:>7} | {lv.height:>7} | {lv.downsample:>10.3f} | {mpp_str:>8}{marker}")

    if report.warning:
        print(f"\n⚠️  {report.warning}")
    elif report.best_level is not None:
        print(f"\n✅ Рекомендований рівень: {report.best_level} (mpp={report.best_level_mpp:.4f}), "
              f"відповідає ~{0.5/report.best_level_mpp*20:.1f}x еквіваленту")
    print(f"{'='*80}")


def collect_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def write_csv(reports: list[SlideReport], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "reader", "base_mpp", "base_mpp_source",
            "best_level", "best_level_mpp", "target_mpp", "warning",
        ])
        for r in reports:
            writer.writerow([
                r.path, r.reader,
                f"{r.base_mpp:.4f}" if r.base_mpp else "",
                r.base_mpp_source,
                r.best_level if r.best_level is not None else "",
                f"{r.best_level_mpp:.4f}" if r.best_level_mpp else "",
                r.target_mpp,
                r.warning or "",
            ])
    log.info(f"CSV звіт збережено: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Шлях до WSI файлу або директорії з WSI")
    parser.add_argument("--target-mpp", type=float, default=0.5,
                         help="Target mpp (default 0.5 = 20x, відповідає pretraining UNI2-h)")
    parser.add_argument("--recursive", action="store_true", help="Рекурсивний пошук у директорії")
    parser.add_argument("--csv", type=Path, default=None, help="Опційно: зберегти зведений звіт у CSV")
    args = parser.parse_args()

    if not args.path.exists():
        log.error(f"Шлях не існує: {args.path}")
        return 1

    files = collect_files(args.path, args.recursive)
    if not files:
        log.error(f"Не знайдено WSI-файлів у {args.path} (розширення: {SUPPORTED_EXTENSIONS})")
        return 1

    log.info(f"Знайдено {len(files)} файл(ів) для перевірки")

    reports = []
    for f in files:
        report = analyze_slide(f, args.target_mpp)
        print_report(report)
        reports.append(report)

    # Зведення: чи всі слайди мають однаковий best_level (важливо для консистентності пайплайну)
    best_levels = {r.best_level for r in reports if r.best_level is not None}
    print(f"\n{'#'*80}")
    if len(best_levels) == 1:
        print(f"# Усі слайди узгоджено використовують level={best_levels.pop()} -- можна хардкодити в pipeline.")
    elif len(best_levels) > 1:
        print(f"# УВАГА: різні слайди потребують різних levels: {sorted(best_levels)}. "
              f"НЕ можна хардкодити один level index для всього датасету -- "
              f"треба обирати level динамічно per-slide (як робить цей скрипт), "
              f"інакше частина слайдів піде в модель з неправильною magnification.")
    else:
        print("# Не вдалося визначити best_level для жодного слайду -- перевір metadata вручну.")
    print(f"{'#'*80}\n")

    if args.csv:
        write_csv(reports, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
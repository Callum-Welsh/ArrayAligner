#!/usr/bin/env python3
"""Detect sufficiently bright peaks and fit them with elliptical 2D Gaussians."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable


import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage, optimize
from skimage.feature import peak_local_max


def ensure_2d(image: np.ndarray) -> np.ndarray:
    """Return the strongest 2D slice for multi-page or multi-channel TIFFs."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        # RGB/RGBA images have the channel axis last.
        if image.shape[-1] in (3, 4) and image.shape[0] > image.shape[-1]:
            return np.mean(image[..., :3], axis=-1)
        # Multi-page stacks often pack the extra axis first.
        return image[0]
    return np.squeeze(image)


def detect_peaks(
    image: np.ndarray,
    min_distance: int,
    threshold_abs: float | None,
    max_peaks: int | None,
) -> np.ndarray:
    """Return the (row, col) coordinates of local maxima."""

    smooth = ndimage.gaussian_filter(image, sigma=1)
    if threshold_abs is None:
        threshold_abs = float(np.mean(smooth) + 1.5 * np.std(smooth))
    num_peaks = np.inf if max_peaks is None or max_peaks <= 0 else max_peaks
    peaks = peak_local_max(
        smooth,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        num_peaks=num_peaks,
        exclude_border=False,
    )
    return peaks


def elliptical_gaussian(params: np.ndarray, coords: np.ndarray) -> np.ndarray:
    (amplitude, xo, yo, sigma_x, sigma_y, theta, offset) = params
    x, y = coords
    x = x - xo
    y = y - yo
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * x + sin_theta * y
    y_rot = -sin_theta * x + cos_theta * y
    gaussian = amplitude * np.exp(
        -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    )
    return gaussian + offset


def fit_gaussian(
    image: np.ndarray,
    peak: np.ndarray,
    patch_radius: int,
    bounds: tuple[np.ndarray, np.ndarray],
    ftol: float,
    xtol: float,
    gtol: float,
) -> dict | None:
    row, col = peak
    y_min = max(row - patch_radius, 0)
    y_max = min(row + patch_radius + 1, image.shape[0])
    x_min = max(col - patch_radius, 0)
    x_max = min(col + patch_radius + 1, image.shape[1])

    patch = image[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None

    y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
    coords = np.vstack((x_grid.ravel(), y_grid.ravel()))
    patch_flat = patch.ravel().astype(float)

    amplitude_guess = float(np.max(patch_flat) - np.min(patch_flat))
    offset_guess = float(np.min(patch_flat))
    guess = np.array([
        amplitude_guess,
        float(col),
        float(row),
        patch.shape[1] / 4
        if patch.shape[1] > 1
        else 1.0,
        patch.shape[0] / 4
        if patch.shape[0] > 1
        else 1.0,
        0.0,
        offset_guess,
    ])

    def residual(params: np.ndarray) -> np.ndarray:
        return elliptical_gaussian(params, coords) - patch_flat

    try:
        result = optimize.least_squares(
            residual,
            guess,
            bounds=bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=200,
        )
    except ValueError:
        return None

    if not result.success:
        return None

    return {
        "row": float(result.x[2]),
        "col": float(result.x[1]),
        "amplitude": float(result.x[0]),
        "sigma_x": float(result.x[3]),
        "sigma_y": float(result.x[4]),
        "theta_deg": float(np.degrees(result.x[5])),
        "offset": float(result.x[6]),
        "residual": float(np.linalg.norm(result.fun)),
    }


def fit_image_array(
    image: np.ndarray,
    patch_radius: int,
    min_distance: int,
    threshold_abs: float | None,
    max_peaks: int | None,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[list[dict], np.ndarray]:
    processed = ensure_2d(image).astype(float)
    peaks = detect_peaks(
        processed,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        max_peaks=max_peaks,
    )

    lower_bounds = np.array([0.0, 0.0, 0.0, 0.1, 0.1, -np.pi, -np.inf])
    upper_bounds = np.array([
        np.inf,
        float(processed.shape[1]),
        float(processed.shape[0]),
        float(patch_radius * 2),
        float(patch_radius * 2),
        np.pi,
        np.inf,
    ])

    results = []
    for peak in peaks:
        fit = fit_gaussian(
            processed,
            peak,
            patch_radius,
            (lower_bounds, upper_bounds),
            ftol,
            xtol,
            gtol,
        )
        if fit is not None:
            results.append(fit)

    return results, processed


def process_image(
    path: pathlib.Path,
    patch_radius: int,
    min_distance: int,
    threshold_abs: float | None,
    max_peaks: int | None,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[list[dict], np.ndarray]:
    raw_image = tifffile.imread(path)
    results, processed = fit_image_array(
        raw_image,
        patch_radius=patch_radius,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        max_peaks=max_peaks,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )
    for fit in results:
        fit["image"] = path.name
    return results, processed


def draw_crosses_on_image(
    image: np.ndarray,
    fits: Iterable[dict],
) -> Image.Image:
    arr = np.array(image, dtype=float)
    min_val = float(np.nanmin(arr)) if arr.size else 0.0
    max_val = float(np.nanmax(arr)) if arr.size else 0.0
    scale = max(max_val - min_val, 1e-6)
    normalized = np.clip((arr - min_val) / scale, 0.0, 1.0)
    uint8 = (normalized * 255).astype(np.uint8)
    rgb_array = np.stack([uint8] * 3, axis=-1)
    height = rgb_array.shape[0]
    width = rgb_array.shape[1]
    img = Image.fromarray(rgb_array, mode="RGB")
    draw = ImageDraw.Draw(img)

    def clamp_point(x: float, y: float) -> tuple[int, int]:
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(width - 1, xi))
        yi = max(0, min(height - 1, yi))
        return xi, yi

    for fit in fits:
        center_x = float(fit["col"])
        center_y = float(fit["row"])
        theta = np.radians(float(fit["theta_deg"]))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        axes = [
            (float(fit["sigma_x"]), np.array([cos_t, sin_t], dtype=float)),
            (float(fit["sigma_y"]), np.array([-sin_t, cos_t], dtype=float)),
        ]
        for sigma, direction in axes:
            if sigma <= 0:
                continue
            delta = 2.0 * sigma * direction
            start = np.array([center_x, center_y], dtype=float) - delta
            end = np.array([center_x, center_y], dtype=float) + delta
            draw.line(
                [*clamp_point(*start), *clamp_point(*end)],
                fill=(0, 255, 0),
                width=1,
            )
    return img


def annotate_image(
    image: np.ndarray,
    fits: Iterable[dict],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_crosses_on_image(image, fits).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Turn TIFFs into NumPy arrays, find bright local maxima, and fit each with an elliptical Gaussian."""
    )
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        default=pathlib.Path("Testing"),
        help="Directory containing .tiff files (default: Testing).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("gaussian_centers.csv"),
        help="CSV path for the fitted results (default: gaussian_centers.csv).",
    )
    parser.add_argument(
        "--patch",
        "-p",
        type=int,
        default=12,
        help="Radius of the square patch (default: 12).",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=8,
        help="Minimum pixel distance between peaks (default: 8).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold for a valid peak (absolute pixel units).",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=10,
        help="Stop after detecting this many peaks per image (default: 10, use 0 or negative for no limit).",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-6,
        help="ftol passed to scipy.optimize.least_squares (default: 1e-6).",
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=1e-6,
        help="xtol passed to scipy.optimize.least_squares (default: 1e-6).",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-6,
        help="gtol passed to scipy.optimize.least_squares (default: 1e-6).",
    )
    parser.add_argument(
        "--annotated-dir",
        type=pathlib.Path,
        default=pathlib.Path("annotated"),
        help="Directory to write annotated overlays (default: annotated).",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip writing annotated overlays even when --annotated-dir is set.",
    )
    return parser.parse_args()


def write_results(path: pathlib.Path, records: Iterable[dict]) -> None:
    fieldnames = [
        "image",
        "row",
        "col",
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta_deg",
        "offset",
        "residual",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    args = parse_args()
    tif_paths = sorted(args.input.glob("*.tif")) + sorted(args.input.glob("*.tiff"))
    if not tif_paths:
        raise SystemExit(f"No TIFF files found in {args.input}")

    max_peaks = None if args.max_peaks is None or args.max_peaks <= 0 else args.max_peaks
    annotated_dir = None if args.no_annotated else args.annotated_dir
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for tif in tif_paths:
        results, image = process_image(
            tif,
            patch_radius=args.patch,
            min_distance=args.min_distance,
            threshold_abs=args.threshold,
            max_peaks=max_peaks,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        )
        all_results.extend(results)
        if annotated_dir is not None:
            annotate_image(
                image,
                results,
                annotated_dir / f"{tif.stem}_annotated.png",
            )

    if not all_results:
        raise SystemExit("No peaks were fitted. Try lowering the threshold or adjusting --patch.")

    write_results(args.output, all_results)
    print(
        f"Fitted {len(all_results)} peaks from {len(tif_paths)} image(s). Results written to {args.output}."
    )


if __name__ == "__main__":
    main()

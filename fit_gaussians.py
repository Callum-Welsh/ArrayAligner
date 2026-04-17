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


def compute_alignment(
    left_fits: list[dict],
    right_fits: list[dict],
    cal_scale: float | None = None,
    anchor_k: int = 0,
    scale_ref_m: int | None = None,
    anchor_k_tweezer: int | None = None,
    scale_ref_m_tweezer: int | None = None,
) -> dict:
    """Compute δ parameters (shift + linear scale) to add to the left array to overlap with right.

    Peaks are sorted top-to-bottom (smallest image row = detected-pair index 0).

    anchor_k           : detected-pair index (0 = topmost) of the anchor peak.
                         Correction is exactly zero at this peak.
    scale_ref_m        : detected-pair index of the scale-reference peak.
                         Defaults to the last (bottom-most) detected pair.
    anchor_k_tweezer   : actual tweezer number of the anchor peak in the full array.
                         Defaults to anchor_k if not given.
    scale_ref_m_tweezer: actual tweezer number of the scale-reference peak.
                         Defaults to scale_ref_m if not given.

    Model (in tweezer space):
        left_y'[t] + shift_y + (t − t_k) · a = right_y'[t]
    where t is the tweezer index, t_k is anchor_k_tweezer, and a is in units/tweezer.

    Returns a dict with keys:
      shift_x_rot, shift_y_rot     – offset at anchor in rotated x'/y' frame (add to left)
      scale_a                      – linear correction per tweezer step in y'
      units                        – 'MHz' if cal_scale provided, else 'px'
      n_pairs                      – number of matched detected pairs
      anchor_k, scale_ref_m        – detected-pair indices actually used (clamped)
      anchor_k_tweezer             – tweezer index of anchor
      scale_ref_m_tweezer          – tweezer index of scale ref
      axis_vector                  – (dx, dy) unit vector used as y' axis
    """
    left_sorted  = sorted(left_fits,  key=lambda f: f["row"])
    right_sorted = sorted(right_fits, key=lambda f: f["row"])
    n = min(len(left_sorted), len(right_sorted))

    # Build rotation axis: average direction from first→last peak in each image
    vectors = []
    for fits in (left_sorted, right_sorted):
        if len(fits) < 2:
            continue
        vec = np.array([fits[-1]["col"] - fits[0]["col"], fits[-1]["row"] - fits[0]["row"]], dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vectors.append(vec / norm)

    if vectors:
        u_y = np.mean(vectors, axis=0)
        u_y /= np.linalg.norm(u_y)
    else:
        u_y = np.array([0.0, 1.0])
    u_x = np.array([u_y[1], -u_y[0]], dtype=float)

    def rot(dx: float, dy: float) -> tuple[float, float]:
        d = np.array([dx, dy])
        return float(d.dot(u_x)), float(d.dot(u_y))

    # dy[i] = left_y'[i] - right_y'[i]  (positive → left is BELOW right)
    pairs = [
        rot(l["col"] - r["col"], l["row"] - r["row"])
        for l, r in zip(left_sorted[:n], right_sorted[:n])
    ]

    # Clamp detected-pair indices to valid range
    k = max(0, min(anchor_k, n - 1))
    m = max(0, min(scale_ref_m if scale_ref_m is not None else n - 1, n - 1))

    # Tweezer indices default to detected-pair indices if not supplied
    k_tw = anchor_k_tweezer    if anchor_k_tweezer    is not None else k
    m_tw = scale_ref_m_tweezer if scale_ref_m_tweezer is not None else m

    dx_k, dy_k = pairs[k]
    _,    dy_m = pairs[m]

    # δ = what to add to the RIGHT array to overlap the left.
    # Coordinate convention: negative y' = up (toward top of image), positive y' = down.
    # Positive a expands the right array; negative a contracts it.
    #
    # Model: right_y'[t] + shift_y + (t − t_k)·a = left_y'[t]
    # At t = t_k:  shift_y = left_y'[t_k] − right_y'[t_k] = dy_k
    # At t = t_m:  a = (dy_m − dy_k) / (t_m − t_k)
    # ── SIGN CONVENTION NOTE ─────────────────────────────────────────────────
    # shift_y = left_y' − right_y'.  Positive y' is DOWN (larger pixel row).
    # Therefore:
    #   shift_y > 0  →  left is BELOW right  (left is at higher frequency)
    #   shift_y < 0  →  right is BELOW left  (right is at higher frequency)
    #
    # If you want "right is Δ MHz above left → positive output", negate these
    # values: the code reports the CORRECTION to add to right to reach left,
    # not the signed frequency offset of right relative to left.
    #
    # ── CALIBRATION (scale) ───────────────────────────────────────────────
    # cal_scale (MHz/px) is computed from the pixel spacing of the top-two
    # right-image peaks and the user-supplied MHz distance between them.
    # It is always positive; the SIGN of the output comes only from shift_y/a.
    shift_x = dx_k    # left_x' − right_x' at anchor
    shift_y = dy_k    # left_y' − right_y' at anchor  (negative → right must move up)
    a = (dy_m - dy_k) / (m_tw - k_tw) if m_tw != k_tw else 0.0

    # ── RESCALING: pixels → physical units ────────────────────────────────
    # Multiply raw pixel shifts by cal_scale (MHz/px) to convert to MHz.
    # If cal_scale is None (no calibration), scale=1.0 and units stay as px.
    scale = cal_scale if cal_scale is not None else 1.0
    units = "MHz" if cal_scale is not None else "px"

    return {
        "shift_x_rot":        shift_x * scale,
        "shift_y_rot":        shift_y * scale,
        "scale_a":            a * scale,
        "units":              units,
        "n_pairs":            n,
        "anchor_k":           k,
        "scale_ref_m":        m,
        "anchor_k_tweezer":   k_tw,
        "scale_ref_m_tweezer": m_tw,
        "axis_vector":        u_y.tolist(),
    }


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
    parser.add_argument(
        "--left",
        type=pathlib.Path,
        default=None,
        help="Left (interlace) TIFF for alignment comparison.",
    )
    parser.add_argument(
        "--right",
        type=pathlib.Path,
        default=None,
        help="Right (main) TIFF for alignment comparison.",
    )
    parser.add_argument(
        "--cal-dist",
        type=float,
        default=None,
        help="Calibration: MHz between the top two adjacent peaks of the right image.",
    )
    parser.add_argument(
        "--anchor-k",
        type=int,
        default=0,
        help=(
            "Detected-pair index of the anchor peak (default: 0 = topmost detected peak). "
            "Peaks sorted top-to-bottom: 0 is highest in image, 1 is next down, etc. "
            "Zero correction is applied at this peak."
        ),
    )
    parser.add_argument(
        "--scale-ref-m",
        type=int,
        default=None,
        help=(
            "Detected-pair index of the scale-reference peak (default: last/bottommost). "
            "Same top-to-bottom ordering as --anchor-k."
        ),
    )
    parser.add_argument(
        "--anchor-k-tweezer",
        type=int,
        default=None,
        help=(
            "Tweezer index (in the full array) of the anchor peak. "
            "Defaults to --anchor-k if not given. "
            "The scale a is computed per tweezer step using this and --scale-ref-m-tweezer."
        ),
    )
    parser.add_argument(
        "--scale-ref-m-tweezer",
        type=int,
        default=None,
        help=(
            "Tweezer index (in the full array) of the scale-reference peak. "
            "Defaults to --scale-ref-m if not given."
        ),
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


def _print_alignment(
    left_fits: list[dict],
    right_fits: list[dict],
    cal_dist: float | None,
    anchor_k: int = 0,
    scale_ref_m: int | None = None,
    anchor_k_tweezer: int | None = None,
    scale_ref_m_tweezer: int | None = None,
) -> None:
    """Compute and print δ parameters; calibrates using top-two-peak spacing of right image."""
    right_sorted = sorted(right_fits, key=lambda f: f["row"])
    cal_scale: float | None = None
    if cal_dist is not None and len(right_sorted) >= 2:
        r0, r1 = right_sorted[0], right_sorted[1]
        px_spacing = float(np.hypot(r1["col"] - r0["col"], r1["row"] - r0["row"]))
        if px_spacing > 1e-6:
            cal_scale = cal_dist / px_spacing

    result = compute_alignment(
        left_fits, right_fits, cal_scale,
        anchor_k, scale_ref_m, anchor_k_tweezer, scale_ref_m_tweezer,
    )
    u    = result["units"]
    k    = result["anchor_k"]
    m    = result["scale_ref_m"]
    k_tw = result["anchor_k_tweezer"]
    m_tw = result["scale_ref_m_tweezer"]
    n    = result["n_pairs"]
    print(
        f"δ (add to RIGHT array to overlap left)\n"
        f"  {n} detected pairs  |  pair indices: 0 = topmost → {n-1} = bottommost\n"
        f"  y' convention: negative = up (toward top of image), positive = down\n"
        f"  scale convention: positive a expands right array, negative a contracts it\n"
        f"\n"
        f"  Anchor   k = pair {k}, tweezer {k_tw}  (zero correction here)\n"
        f"  Scale ref m = pair {m}, tweezer {m_tw}  (pins the scale)\n"
        f"\n"
        f"  Δy' = {result['shift_y_rot']:+.6f} {u}\n"
        f"  Δx' = {result['shift_x_rot']:+.6f} {u}\n"
        f"  a   = {result['scale_a']:+.6f} {u}/tweezer\n"
        f"\n"
        f"  right_y'[t] + ({result['shift_y_rot']:+.6f}) + (t − {k_tw})·({result['scale_a']:+.6f}) = left_y'[t]\n"
        f"  where t = tweezer index"
    )
    if cal_dist is not None and cal_scale is None:
        print("  Warning: calibration requires at least 2 peaks in the right image.")


def main() -> None:
    args = parse_args()

    fit_kwargs = dict(
        patch_radius=args.patch,
        min_distance=args.min_distance,
        threshold_abs=args.threshold,
        max_peaks=None if args.max_peaks is None or args.max_peaks <= 0 else args.max_peaks,
        ftol=args.ftol,
        xtol=args.xtol,
        gtol=args.gtol,
    )

    # Comparison mode: --left and --right provided directly
    if args.left is not None and args.right is not None:
        left_fits,  left_img  = process_image(args.left,  **fit_kwargs)
        right_fits, right_img = process_image(args.right, **fit_kwargs)
        for f in left_fits:  f["image"] = args.left.name
        for f in right_fits: f["image"] = args.right.name
        write_results(args.output, left_fits + right_fits)
        print(f"Fitted {len(left_fits)} left + {len(right_fits)} right peaks. Results written to {args.output}.")
        _print_alignment(
            left_fits, right_fits, args.cal_dist,
            args.anchor_k, args.scale_ref_m,
            args.anchor_k_tweezer, args.scale_ref_m_tweezer,
        )
        annotated_dir = None if args.no_annotated else args.annotated_dir
        if annotated_dir is not None:
            annotated_dir.mkdir(parents=True, exist_ok=True)
            for path, fits, img in ((args.left, left_fits, left_img), (args.right, right_fits, right_img)):
                annotate_image(img, fits, annotated_dir / f"{path.stem}_annotated.png")
        return

    # Batch mode: process all TIFFs in --input directory
    tif_paths = sorted(args.input.glob("*.tif")) + sorted(args.input.glob("*.tiff"))
    if not tif_paths:
        raise SystemExit(f"No TIFF files found in {args.input}")

    annotated_dir = None if args.no_annotated else args.annotated_dir
    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for tif in tif_paths:
        results, image = process_image(tif, **fit_kwargs)
        all_results.extend(results)
        if annotated_dir is not None:
            annotate_image(image, results, annotated_dir / f"{tif.stem}_annotated.png")

    if not all_results:
        raise SystemExit("No peaks were fitted. Try lowering the threshold or adjusting --patch.")

    write_results(args.output, all_results)
    print(f"Fitted {len(all_results)} peaks from {len(tif_paths)} image(s). Results written to {args.output}.")


if __name__ == "__main__":
    main()

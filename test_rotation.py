#!/usr/bin/env python3
"""
Rotation sanity-check: create a synthetic image with Gaussian peaks along
a known diagonal, run the fitting + rotation logic, and save an annotated
PNG so the coordinate frame can be verified visually.

Expected result
---------------
The peaks are placed along a line tilted ~20 degrees clockwise from vertical
(i.e. bottom peak is to the right of the top peak).  The saved image should
show:
  - green cross overlays centred on each fitted peak
  - a blue line through all peaks (the computed y' axis / rotation axis)
  - a red arrow at the anchor peak showing the x' direction (perpendicular)
  - cyan labels: "y' +" points DOWN along the array, "x' +" points to the
    right of the array direction.

If the axes are drawn where you expect, rotation is working correctly.

Usage
-----
  python test_rotation.py                 # saves Testing/rotation_test.png
  python test_rotation.py --show          # also opens the image
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fit_gaussians import compute_alignment, draw_crosses_on_image, fit_image_array

# ── synthetic image parameters ────────────────────────────────────────────────
IMG_H, IMG_W = 400, 400
N_PEAKS = 6
PEAK_AMPLITUDE = 5000.0
PEAK_SIGMA = 6.0
BACKGROUND = 100.0

# Angle of the peak array from vertical, in degrees clockwise.
# 0° = perfectly vertical column; +20° = bottom peak shifted right.
ARRAY_ANGLE_DEG = 20.0

# Spacing between adjacent peaks along the array axis, in pixels.
PEAK_SPACING_PX = 55.0


def make_synthetic_image(
    height: int,
    width: int,
    peak_positions: list[tuple[float, float]],
    amplitude: float,
    sigma: float,
    background: float,
) -> np.ndarray:
    img = np.full((height, width), background, dtype=float)
    rows, cols = np.mgrid[0:height, 0:width]
    for r, c in peak_positions:
        img += amplitude * np.exp(-0.5 * (((cols - c) / sigma) ** 2 + ((rows - r) / sigma) ** 2))
    return img


def peak_positions_along_axis(
    n: int,
    spacing: float,
    angle_deg: float,
    img_h: int,
    img_w: int,
) -> list[tuple[float, float]]:
    """Return (row, col) positions for n peaks evenly spaced along a tilted line."""
    angle_rad = math.radians(angle_deg)
    # Unit vector pointing DOWN along the array (y' axis direction in image coords):
    #   row increases downward, col increases rightward.
    #   angle_deg=0 → pure vertical (Δcol=0, Δrow=spacing)
    #   angle_deg>0 → clockwise tilt (Δcol>0)
    d_row = math.cos(angle_rad)
    d_col = math.sin(angle_rad)

    center_r = img_h / 2.0
    center_c = img_w / 2.0
    half_len = (n - 1) / 2.0 * spacing

    positions = []
    for i in range(n):
        t = (i - half_len / spacing) * spacing
        positions.append((center_r + t * d_row, center_c + t * d_col))
    return positions


def draw_axis_overlay(
    pil_img: Image.Image,
    axis_vector: list[float],
    anchor_row: float,
    anchor_col: float,
    arrow_len: int = 60,
) -> Image.Image:
    """Draw the y' axis (blue) and x' direction (red arrow) on a copy of pil_img."""
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = img.height, img.width

    u_y = np.array(axis_vector, dtype=float)
    u_x = np.array([u_y[1], -u_y[0]], dtype=float)  # perpendicular, points "right"

    # Blue line: y' axis through centre of image
    cx, cy = w / 2.0, h / 2.0
    scale = max(h, w) * 0.6
    p1 = (cx - u_y[0] * scale, cy - u_y[1] * scale)
    p2 = (cx + u_y[0] * scale, cy + u_y[1] * scale)
    draw.line([p1[0], p1[1], p2[0], p2[1]], fill=(0, 80, 255), width=2)

    # Red arrow: x' direction from anchor peak
    ax, ay = anchor_col, anchor_row
    tip_x = ax + u_x[0] * arrow_len
    tip_y = ay + u_x[1] * arrow_len
    draw.line([ax, ay, tip_x, tip_y], fill=(255, 50, 50), width=3)

    # Labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label_y_plus = (
        int(cx + u_y[0] * (scale * 0.85)),
        int(cy + u_y[1] * (scale * 0.85)),
    )
    label_x_plus = (int(tip_x + 4), int(tip_y))

    draw.text(label_y_plus, "y'+ (down array)", fill=(0, 180, 255), font=font)
    draw.text(label_x_plus, "x'+ (right of array)", fill=(255, 80, 80), font=font)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotation sanity-check for ImageAligner.")
    parser.add_argument("--show", action="store_true", help="Open the output image after saving.")
    args = parser.parse_args()

    out_dir = pathlib.Path("Testing")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rotation_test.png"

    # ── Build synthetic image ─────────────────────────────────────────────
    positions = peak_positions_along_axis(N_PEAKS, PEAK_SPACING_PX, ARRAY_ANGLE_DEG, IMG_H, IMG_W)
    img_array = make_synthetic_image(IMG_H, IMG_W, positions, PEAK_AMPLITUDE, PEAK_SIGMA, BACKGROUND)

    # Add a tiny bit of noise so detect_peaks has a clear threshold
    rng = np.random.default_rng(42)
    img_array += rng.normal(0, 10, img_array.shape)

    print(f"Synthetic image: {N_PEAKS} peaks, array angle {ARRAY_ANGLE_DEG}° from vertical")
    print(f"True peak positions (row, col):")
    for i, (r, c) in enumerate(positions):
        print(f"  peak {i}: row={r:.1f}  col={c:.1f}")

    # ── Fit peaks ─────────────────────────────────────────────────────────
    fits, processed = fit_image_array(
        img_array,
        patch_radius=15,
        min_distance=20,
        threshold_abs=BACKGROUND + PEAK_AMPLITUDE * 0.3,
        max_peaks=N_PEAKS + 2,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    fits_sorted = sorted(fits, key=lambda f: f["row"])
    print(f"\nFitted {len(fits_sorted)} peaks:")
    for i, f in enumerate(fits_sorted):
        print(f"  peak {i}: row={f['row']:.2f}  col={f['col']:.2f}")

    # ── Compute rotation axis (reuse compute_alignment with image = both sides) ──
    # Pass the same fits as both left and right so alignment offsets are all 0,
    # but the rotation axis is derived from the peak array direction.
    al = compute_alignment(fits_sorted, fits_sorted, cal_scale=None, anchor_k=0)
    axis_vec = al["axis_vector"]

    # Expected axis vector from the known angle
    angle_rad = math.radians(ARRAY_ANGLE_DEG)
    expected_u_y = [math.sin(angle_rad), math.cos(angle_rad)]  # [dx, dy] in image coords
    dot = abs(np.dot(axis_vec, expected_u_y))
    angle_error_deg = math.degrees(math.acos(min(dot, 1.0)))

    print(f"\nComputed axis vector (dx, dy): [{axis_vec[0]:.4f}, {axis_vec[1]:.4f}]")
    print(f"Expected axis vector (dx, dy): [{expected_u_y[0]:.4f}, {expected_u_y[1]:.4f}]")
    print(f"Angular error: {angle_error_deg:.3f}°  (should be < 1°)")

    assert angle_error_deg < 2.0, (
        f"Rotation axis error {angle_error_deg:.2f}° exceeds 2° tolerance — "
        "rotation logic may be broken."
    )
    print("PASS: rotation axis matches expected direction within tolerance.")

    # Sign convention check: shift values should be ~0 when same fits used for both sides
    print(f"\nSign convention (shift should be ~0 when left==right):")
    print(f"  shift_y = {al['shift_y_rot']:.6f} px  (expected ~0)")
    print(f"  shift_x = {al['shift_x_rot']:.6f} px  (expected ~0)")

    # ── Save annotated image ──────────────────────────────────────────────
    base_img = draw_crosses_on_image(processed, fits_sorted)

    anchor_fit = fits_sorted[0]
    annotated = draw_axis_overlay(
        base_img,
        axis_vec,
        anchor_row=anchor_fit["row"],
        anchor_col=anchor_fit["col"],
    )

    annotated.save(out_path)
    print(f"\nSaved annotated image → {out_path}")
    print("Visual check: blue line = y' axis (should run along the peak array),")
    print("              red arrow = x'+ direction (perpendicular, pointing right of the array).")
    print("              y'+ points DOWN the array (larger pixel row = larger y' = higher index).")

    if args.show:
        annotated.show()


if __name__ == "__main__":
    main()

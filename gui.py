#!/usr/bin/env python3
"""Tkinter GUI to load two TIFFs, fit peaks, and show cross overlays."""

from __future__ import annotations

import csv
import math
import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import tifffile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from fit_gaussians import draw_crosses_on_image, ensure_2d, fit_image_array


class GaussianFitApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Gaussian Peak Viewer")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.max_display_size = (int(screen_width * 0.4), int(screen_height * 0.4))
        self.images: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.fits: dict[str, list[dict]] = {"left": [], "right": []}
        self.paths: dict[str, pathlib.Path | None] = {"left": None, "right": None}
        self.param_vars: dict[str, tk.StringVar] = {}
        self.left_label: ttk.Label
        self.right_label: ttk.Label
        self.status_var = tk.StringVar(value="Load two TIFFs and click Fit")
        self.axis_mode_var = tk.StringVar(value="original")
        self.results_status_var = tk.StringVar(value="Distances will appear after fitting.")
        self.shift_info_var = tk.StringVar(value="Shift info will appear after matching.")
        self.distance_pairs: list[dict] = []
        self.calibration_distance_var = tk.StringVar(value="1")
        self.calibration_scale: float | None = None
        self.number_tweezers_var = tk.StringVar(value="37")
        self.last_axis_label = "dx/dy"
        self.last_axis_key_x = "dx"
        self.last_axis_key_y = "dy"
        self.figure = Figure(figsize=(6, 4))
        self.ax_euclid = self.figure.add_subplot(2, 1, 1)
        self.ax_dxdy = self.figure.add_subplot(2, 1, 2)
        self.canvas: FigureCanvasTkAgg | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=8)

        entries = [
            ("Patch", "patch", "12"),
            ("Min distance", "min_distance", "8"),
            ("Threshold", "threshold", ""),
            ("Max peaks", "max_peaks", "10"),
            ("ftol", "ftol", "1e-6"),
            ("xtol", "xtol", "1e-6"),
            ("gtol", "gtol", "1e-6"),
        ]
        for label_text, key, default in entries:
            self.param_vars[key] = tk.StringVar(value=default)
            frame = ttk.Frame(control_frame)
            frame.pack(side=tk.LEFT, padx=4)
            ttk.Label(frame, text=f"{label_text}:").pack(anchor=tk.W)
            ttk.Entry(frame, width=7, textvariable=self.param_vars[key]).pack()

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10)
        ttk.Button(button_frame, text="Load left TIFF", command=lambda: self._load_image("left")).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(button_frame, text="Load right TIFF", command=lambda: self._load_image("right")).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Fit", command=self._run_fit).pack(side=tk.LEFT, padx=12)
        ttk.Label(button_frame, text="Cal dist (MHz):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(button_frame, width=7, textvariable=self.calibration_distance_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Calibrate", command=self._calibrate).pack(side=tk.LEFT, padx=4)
        ttk.Label(button_frame, text="Number of Tweezers:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(button_frame, width=5, textvariable=self.number_tweezers_var).pack(side=tk.LEFT, padx=(0, 12))

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        images_tab = ttk.Frame(notebook)
        results_tab = ttk.Frame(notebook)
        notebook.add(images_tab, text="Images")
        notebook.add(results_tab, text="Results")

        display_frame = ttk.Frame(images_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.LabelFrame(display_frame, text="Left Image")
        left_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4)
        self.left_label = ttk.Label(left_panel, text="No image loaded", anchor=tk.CENTER)
        self.left_label.pack(expand=True, fill=tk.BOTH)
        self.left_name_label = ttk.Label(left_panel, text="")
        self.left_name_label.pack(fill=tk.X)

        right_panel = ttk.LabelFrame(display_frame, text="Right Image")
        right_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4)
        self.right_label = ttk.Label(right_panel, text="No image loaded", anchor=tk.CENTER)
        self.right_label.pack(expand=True, fill=tk.BOTH)
        self.right_name_label = ttk.Label(right_panel, text="")
        self.right_name_label.pack(fill=tk.X)

        result_canvas_frame = ttk.Frame(results_tab)
        result_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        axis_mode_frame = ttk.Frame(results_tab)
        axis_mode_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(axis_mode_frame, text="Axis view:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            axis_mode_frame,
            text="Original dx/dy",
            variable=self.axis_mode_var,
            value="original",
            command=self._update_results,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(
            axis_mode_frame,
            text="Rotated x'/y'",
            variable=self.axis_mode_var,
            value="rotated",
            command=self._update_results,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Label(results_tab, textvariable=self.results_status_var).pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(results_tab, textvariable=self.shift_info_var).pack(fill=tk.X, padx=4, pady=2)
        result_button_frame = ttk.Frame(results_tab)
        result_button_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(result_button_frame, text="Save distances CSV", command=self._save_results).pack(side=tk.LEFT)

        ttk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=10, pady=4)

    def _load_image(self, side: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select {side} image",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*")] ,
        )
        if not path:
            return
        try:
            raw = tifffile.imread(path)
        except Exception as exc:  # pragma: no cover - external I/O
            messagebox.showerror("Failed to load image", str(exc))
            return
        self.images[side] = ensure_2d(raw)
        self.paths[side] = pathlib.Path(path)
        self.fits[side] = []
        self.calibration_scale = None
        self._update_display(side)
        self._update_name(side)
        self._update_status()
        self._update_results()

    def _update_name(self, side: str) -> None:
        name = self.paths[side].name if self.paths[side] else ""
        label = self.left_name_label if side == "left" else self.right_name_label
        label.config(text=name)

    def _update_display(self, side: str) -> None:
        image = self.images[side]
        label = self.left_label if side == "left" else self.right_label
        if image is None:
            label.config(text="No image loaded", image="")
            return
        overlay = draw_crosses_on_image(image, self.fits[side])
        display_image = overlay.copy()
        display_image.thumbnail(self.max_display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        label.config(image=photo, text="")
        label.image = photo

    def _run_fit(self) -> None:
        if self.images["left"] is None or self.images["right"] is None:
            messagebox.showwarning("Missing images", "Load both left and right images before fitting.")
            return
        self.calibration_scale = None
        try:
            patch = int(self.param_vars["patch"].get())
            min_distance = int(self.param_vars["min_distance"].get())
            threshold_str = self.param_vars["threshold"].get().strip()
            threshold = float(threshold_str) if threshold_str else None
            max_peaks = int(self.param_vars["max_peaks"].get())
            max_peaks = None if max_peaks <= 0 else max_peaks
            ftol = float(self.param_vars["ftol"].get())
            xtol = float(self.param_vars["xtol"].get())
            gtol = float(self.param_vars["gtol"].get())
        except ValueError as exc:
            messagebox.showerror("Invalid parameter", str(exc))
            return
        for side in ("left", "right"):
            results, processed = fit_image_array(
                self.images[side],
                patch_radius=patch,
                min_distance=min_distance,
                threshold_abs=threshold,
                max_peaks=max_peaks,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
            self.fits[side] = results
            self.images[side] = processed
            self._update_display(side)
        self._update_status()
        self._update_results()
        total = len(self.fits["left"]) + len(self.fits["right"])
        messagebox.showinfo("Fit complete", f"Fitted {total} peaks total.")

    def _calibrate(self) -> None:
        if len(self.fits["right"]) < 2:
            messagebox.showwarning("Not enough peaks", "Fit the right image first so that at least two peaks are available.")
            return
        right = sorted(self.fits["right"], key=lambda fit: fit["row"])
        first, second = right[0], right[1]
        dx = float(first["col"]) - float(second["col"])
        dy = float(first["row"]) - float(second["row"])
        measured = math.hypot(dx, dy)
        if measured <= 0:
            messagebox.showerror("Invalid spacing", "The top two peaks are overlapping; cannot calibrate.")
            return
        try:
            target = float(self.calibration_distance_var.get())
        except ValueError:
            messagebox.showerror("Invalid calibration distance", "Enter a numeric value for the calibration distance.")
            return
        if target <= 0:
            messagebox.showerror("Invalid calibration distance", "Calibration distance must be positive.")
            return
        self.calibration_scale = target / measured
        self._update_results()
        messagebox.showinfo(
            "Calibrated",
            f"Calibrated scale: {self.calibration_scale:.6f} MHz/px ({target:.4f} MHz = {measured:.2f} px).",
        )

    def _update_status(self) -> None:
        counts = (
            len(self.fits["left"]) if self.images["left"] is not None else 0,
            len(self.fits["right"]) if self.images["right"] is not None else 0,
        )
        self.status_var.set(f"Left: {counts[0]} peaks · Right: {counts[1]} peaks")

    def _pair_peaks(self, axis_vector: np.ndarray | None = None) -> list[dict]:
        left = sorted(self.fits["left"], key=lambda fit: fit["row"])
        right = sorted(self.fits["right"], key=lambda fit: fit["row"])
        pairs: list[dict] = []
        for idx, (l, r) in enumerate(zip(left, right), start=1):
            dx = float(l["col"]) - float(r["col"])
            dy = float(l["row"]) - float(r["row"])
            pairs.append(
                {
                    "index": idx,
                    "left_row": float(l["row"]),
                    "right_row": float(r["row"]),
                    "dx": dx,
                    "dy": dy,
                    "euclid": math.hypot(dx, dy),
                }
            )
        if axis_vector is not None:
            u_y = axis_vector
            u_x = np.array([u_y[1], -u_y[0]], dtype=float)
            for pair in pairs:
                diff = np.array([pair["dx"], pair["dy"]], dtype=float)
                pair["dx_rot"] = float(diff.dot(u_x))
                pair["dy_rot"] = float(diff.dot(u_y))
                pair["euclid_rot"] = math.hypot(pair["dx_rot"], pair["dy_rot"])
        return pairs

    def _rotation_axis_vector(self) -> np.ndarray | None:
        vectors: list[np.ndarray] = []
        for side in ("left", "right"):
            fits = sorted(self.fits[side], key=lambda fit: fit["row"])
            if len(fits) < 2:
                continue
            first = fits[0]
            last = fits[-1]
            vec = np.array(
                [float(last["col"]) - float(first["col"]), float(last["row"]) - float(first["row"])],
                dtype=float,
            )
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vectors.append(vec / norm)
        if not vectors:
            return None
        avg = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-6:
            return None
        return avg / norm

    def _rotate_diff(self, dx: float, dy: float, axis_vector: np.ndarray) -> tuple[float, float]:
        u_y = axis_vector
        u_x = np.array([u_y[1], -u_y[0]], dtype=float)
        diff = np.array([dx, dy], dtype=float)
        return float(diff.dot(u_x)), float(diff.dot(u_y))

    def _update_results(self) -> None:
        if self.canvas is None:
            return
        axis_mode = self.axis_mode_var.get()
        axis_vector_base = self._rotation_axis_vector()
        axis_vector_plot = axis_vector_base if axis_mode == "rotated" and axis_vector_base is not None else None
        effective_mode = axis_mode
        axis_warning = ""
        if axis_mode == "rotated" and axis_vector_plot is None:
            effective_mode = "original"
            axis_warning = " Rotated axis needs two extremes per image; showing original dx/dy."

        pairs = self._pair_peaks(axis_vector_plot)
        self.last_axis_label = "x'/y'" if effective_mode == "rotated" else "dx/dy"
        self.last_axis_key_x = "dx_rot" if effective_mode == "rotated" else "dx"
        self.last_axis_key_y = "dy_rot" if effective_mode == "rotated" else "dy"

        self.distance_pairs = pairs
        self.ax_euclid.clear()
        self.ax_dxdy.clear()
        scale = self.calibration_scale if self.calibration_scale is not None else 1.0
        units = "MHz" if self.calibration_scale is not None else "px"
        if pairs:
            indices = [pair["index"] for pair in pairs]
            for pair in pairs:
                dx_val = pair.get(self.last_axis_key_x, pair["dx"])
                dy_val = pair.get(self.last_axis_key_y, pair["dy"])
                pair["dx_scaled"] = dx_val * scale
                pair["dy_scaled"] = dy_val * scale
                pair["euclid_scaled"] = pair["euclid"] * scale
            self.ax_euclid.plot(indices, [pair["euclid_scaled"] for pair in pairs], marker="o")
            self.ax_euclid.set_title(f"Euclidean distance ({self.last_axis_label}, by peak order)")
            self.ax_euclid.set_ylabel(f"Euclidean {units}")
            x_label, y_label = self.last_axis_label.split("/")
            self.ax_dxdy.plot(indices, [pair["dx_scaled"] for pair in pairs], label=f"{x_label} ({units})")
            self.ax_dxdy.plot(indices, [pair["dy_scaled"] for pair in pairs], label=f"{y_label} ({units})")
            self.ax_dxdy.set_ylabel(f"Delta ({units})")
            self.ax_dxdy.set_xlabel("Pair index (sorted by row)")
            self.ax_dxdy.legend()
            status = f"{len(pairs)} matched pairs plotted ({self.last_axis_label}, {units})"
            if self.calibration_scale is not None:
                status += f" at {self.calibration_scale:.6f} MHz/px"
            self.results_status_var.set(status + axis_warning)
        else:
            self.ax_euclid.text(
                0.5,
                0.5,
                "Need fits for both images to plot distances",
                ha="center",
                va="center",
                transform=self.ax_euclid.transAxes,
            )
            self.ax_dxdy.text(
                0.5,
                0.5,
                "Run the fit on both sides first",
                ha="center",
                va="center",
                transform=self.ax_dxdy.transAxes,
            )
            self.results_status_var.set("No matched pairs available yet.")

        shift_info = "Shift info will appear after matching."
        if pairs:
            top = pairs[0]
            if axis_vector_base is not None:
                dx_top, dy_top = self._rotate_diff(top["dx"], top["dy"], axis_vector_base)
                axis_hint = "x'/y'"
            else:
                dx_top = top["dx"]
                dy_top = top["dy"]
                axis_hint = "dx/dy"
            shift_x = -dx_top
            shift_y = -dy_top
            shift_x_scaled = shift_x * scale
            shift_y_scaled = shift_y * scale
            try:
                N = int(self.number_tweezers_var.get())
                if N <= 0:
                    raise ValueError
            except ValueError:
                N = 37
                self.number_tweezers_var.set(str(N))
            a_value = shift_y_scaled / N
            shift_info = (
                f"Top-pair shift ({axis_hint}): x'={shift_x_scaled:.3f}{units}, "
                f"y'={shift_y_scaled:.3f}{units}. Raw px shift: ({shift_x:.3f}, {shift_y:.3f}). "
                f"Scale y''=y'+{a_value:.6f}·N ({N} tweezers)."
            )
        self.shift_info_var.set(shift_info)
        self.canvas.draw()

    def _save_results(self) -> None:
        if not self.distance_pairs:
            messagebox.showwarning("No distances", "Run the fit on both images before saving distances.")
            return
        path = filedialog.asksaveasfilename(
            title="Save distances CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
            initialfile="distances.csv",
        )
        if not path:
            return
        with open(path, "w", newline="") as handle:
            fieldnames = [
                "index",
                "left_row",
                "right_row",
                "dx_px",
                "dy_px",
                "euclid_px",
                "dx_axis",
                "dy_axis",
                "axis_label",
                "dx_scaled",
                "dy_scaled",
                "euclid_scaled",
                "units",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            units = "MHz" if self.calibration_scale is not None else "px"
            for record in self.distance_pairs:
                dx_axis = record.get(self.last_axis_key_x, record["dx"])
                dy_axis = record.get(self.last_axis_key_y, record["dy"])
                writer.writerow(
                    {
                        "index": record["index"],
                        "left_row": record["left_row"],
                        "right_row": record["right_row"],
                        "dx_px": record["dx"],
                        "dy_px": record["dy"],
                        "euclid_px": record["euclid"],
                        "dx_axis": dx_axis,
                        "dy_axis": dy_axis,
                        "axis_label": self.last_axis_label,
                        "dx_scaled": record.get("dx_scaled", ""),
                        "dy_scaled": record.get("dy_scaled", ""),
                        "euclid_scaled": record.get("euclid_scaled", ""),
                        "units": units,
                    }
                )
        messagebox.showinfo("Saved", f"Wrote {len(self.distance_pairs)} rows to {path}")


if __name__ == "__main__":
    GaussianFitApp().mainloop()

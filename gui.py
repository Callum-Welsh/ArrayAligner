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

from fit_gaussians import compute_alignment, draw_crosses_on_image, elliptical_gaussian, ensure_2d, fit_image_array


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
        # Detected-pair indices (0 = topmost detected peak, counting downward)
        self.anchor_k_var = tk.StringVar(value="0")
        self.scale_ref_m_var = tk.StringVar(value="9")
        # Corresponding tweezer indices in the full array (default = same as pair indices)
        self.anchor_k_tweezer_var = tk.StringVar(value="0")
        self.scale_ref_m_tweezer_var = tk.StringVar(value="36")
        # StringVars for the big red result display
        self.delta_values_var  = tk.StringVar(value="")
        self.delta_formula_var = tk.StringVar(value="")
        self.delta_info_var    = tk.StringVar(value="Fit both images to see δ results.")
        self.last_axis_label = "dx/dy"
        self.last_axis_key_x = "dx"
        self.last_axis_key_y = "dy"
        self.figure = Figure(figsize=(6, 4))
        self.ax_euclid = self.figure.add_subplot(2, 1, 1)
        self.ax_dxdy = self.figure.add_subplot(2, 1, 2)
        self.canvas: FigureCanvasTkAgg | None = None
        self.fit_tk_canvas: tk.Canvas | None = None
        self.fit_inner_frame: ttk.Frame | None = None
        self._fit_window_id: int | None = None
        self._fit_mpl_canvas: FigureCanvasTkAgg | None = None
        self.overlap_tk_canvas: tk.Canvas | None = None
        self.overlap_inner_frame: ttk.Frame | None = None
        self._overlap_window_id: int | None = None
        self._overlap_mpl_canvas: FigureCanvasTkAgg | None = None
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

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        images_tab = ttk.Frame(notebook)
        results_tab = ttk.Frame(notebook)
        fits_tab = ttk.Frame(notebook)
        overlap_tab = ttk.Frame(notebook)
        notebook.add(images_tab, text="Images")
        notebook.add(results_tab, text="Results")
        notebook.add(fits_tab, text="Fits & Residuals")
        notebook.add(overlap_tab, text="Overlap")

        # ── Overlap tab (scrollable, same pattern as Fits & Residuals) ───────
        overlap_scroll_outer = ttk.Frame(overlap_tab)
        overlap_scroll_outer.pack(fill=tk.BOTH, expand=True)
        overlap_scroll_outer.columnconfigure(0, weight=1)
        overlap_scroll_outer.rowconfigure(0, weight=1)

        self.overlap_tk_canvas = tk.Canvas(overlap_scroll_outer, highlightthickness=0)
        self.overlap_tk_canvas.grid(row=0, column=0, sticky="nsew")

        overlap_vbar = ttk.Scrollbar(overlap_scroll_outer, orient=tk.VERTICAL,
                                     command=self.overlap_tk_canvas.yview)
        overlap_vbar.grid(row=0, column=1, sticky="ns")
        self.overlap_tk_canvas.configure(yscrollcommand=overlap_vbar.set)

        self.overlap_inner_frame = ttk.Frame(self.overlap_tk_canvas)
        self._overlap_window_id = self.overlap_tk_canvas.create_window(
            (0, 0), window=self.overlap_inner_frame, anchor="nw"
        )

        def _on_overlap_inner_configure(event: tk.Event) -> None:
            self.overlap_tk_canvas.configure(
                scrollregion=self.overlap_tk_canvas.bbox("all")
            )

        def _on_overlap_canvas_resize(event: tk.Event) -> None:
            if self._overlap_window_id is not None:
                self.overlap_tk_canvas.itemconfig(
                    self._overlap_window_id, width=event.width
                )

        def _on_overlap_mousewheel(event: tk.Event) -> None:
            self.overlap_tk_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            )

        self.overlap_inner_frame.bind("<Configure>", _on_overlap_inner_configure)
        self.overlap_tk_canvas.bind("<Configure>", _on_overlap_canvas_resize)
        self.overlap_tk_canvas.bind("<MouseWheel>", _on_overlap_mousewheel)
        self.overlap_inner_frame.bind("<MouseWheel>", _on_overlap_mousewheel)

        # ── Fits & Residuals tab ─────────────────────────────────────────────
        fits_scroll_outer = ttk.Frame(fits_tab)
        fits_scroll_outer.pack(fill=tk.BOTH, expand=True)
        fits_scroll_outer.columnconfigure(0, weight=1)
        fits_scroll_outer.rowconfigure(0, weight=1)

        self.fit_tk_canvas = tk.Canvas(fits_scroll_outer, highlightthickness=0)
        self.fit_tk_canvas.grid(row=0, column=0, sticky="nsew")

        fits_vbar = ttk.Scrollbar(fits_scroll_outer, orient=tk.VERTICAL,
                                  command=self.fit_tk_canvas.yview)
        fits_vbar.grid(row=0, column=1, sticky="ns")
        self.fit_tk_canvas.configure(yscrollcommand=fits_vbar.set)

        self.fit_inner_frame = ttk.Frame(self.fit_tk_canvas)
        self._fit_window_id = self.fit_tk_canvas.create_window(
            (0, 0), window=self.fit_inner_frame, anchor="nw"
        )

        def _on_fit_inner_configure(event: tk.Event) -> None:
            self.fit_tk_canvas.configure(
                scrollregion=self.fit_tk_canvas.bbox("all")
            )

        def _on_fit_canvas_resize(event: tk.Event) -> None:
            if self._fit_window_id is not None:
                self.fit_tk_canvas.itemconfig(
                    self._fit_window_id, width=event.width
                )

        def _on_fit_mousewheel(event: tk.Event) -> None:
            self.fit_tk_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            )

        self.fit_inner_frame.bind("<Configure>", _on_fit_inner_configure)
        self.fit_tk_canvas.bind("<Configure>", _on_fit_canvas_resize)
        self.fit_tk_canvas.bind("<MouseWheel>", _on_fit_mousewheel)
        self.fit_inner_frame.bind("<MouseWheel>", _on_fit_mousewheel)

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

        # ── δ alignment reference inputs ─────────────────────────────────────────
        # All peak indices count from 0 = topmost detected peak, downward.
        align_frame = ttk.LabelFrame(
            results_tab,
            text="δ alignment references  —  all indices: 0 = topmost detected peak, counting downward",
        )
        align_frame.pack(fill=tk.X, padx=4, pady=4)

        # ── k (anchor) ──────────────────────────────
        k_outer = ttk.LabelFrame(align_frame, text="k  —  ANCHOR  (zero correction at this peak)")
        k_outer.pack(side=tk.LEFT, padx=10, pady=6, fill=tk.Y)
        ttk.Label(k_outer, text="Detected-pair index\n(0 = topmost)").pack(pady=(4, 0))
        ttk.Entry(k_outer, width=6, textvariable=self.anchor_k_var).pack(pady=2)
        ttk.Label(k_outer, text="Tweezer index\n(in full array)").pack(pady=(6, 0))
        ttk.Entry(k_outer, width=6, textvariable=self.anchor_k_tweezer_var).pack(pady=2)

        ttk.Separator(align_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

        # ── m (scale ref) ───────────────────────────
        m_outer = ttk.LabelFrame(align_frame, text="m  —  SCALE REFERENCE  (pins the scale a)")
        m_outer.pack(side=tk.LEFT, padx=10, pady=6, fill=tk.Y)
        ttk.Label(m_outer, text="Detected-pair index\n(0 = topmost)").pack(pady=(4, 0))
        ttk.Entry(m_outer, width=6, textvariable=self.scale_ref_m_var).pack(pady=2)
        ttk.Label(m_outer, text="Tweezer index\n(in full array)").pack(pady=(6, 0))
        ttk.Entry(m_outer, width=6, textvariable=self.scale_ref_m_tweezer_var).pack(pady=2)

        ttk.Separator(align_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

        right_side = ttk.Frame(align_frame)
        right_side.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.BOTH, expand=True)
        ttk.Label(
            right_side,
            text=(
                "Model (t = tweezer index):\n"
                "right_y'[t]  +  Δy'  +  (t − t_k) · a  =  left_y'[t]\n"
                "\n"
                "negative y' = up    positive y' = down\n"
                "positive a = expand right array\n"
                "negative a = contract right array"
            ),
            font=("TkFixedFont", 10),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        ttk.Button(right_side, text="Recalculate δ", command=self._update_results).pack(
            anchor=tk.W, pady=(8, 0)
        )
        # ────────────────────────────────────────────────────────────────────────

        # ── big red δ results display ────────────────────────────────────────────
        BIG_RED  = ("TkFixedFont", 14, "bold")
        INFO_FONT = ("TkDefaultFont", 10)
        delta_frame = ttk.LabelFrame(results_tab, text="δ results  (add these to the RIGHT array to overlap left)")
        delta_frame.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(
            delta_frame, textvariable=self.delta_values_var,
            font=BIG_RED, fg="red", justify=tk.LEFT, anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=(8, 4))

        ttk.Separator(delta_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        tk.Label(
            delta_frame, textvariable=self.delta_formula_var,
            font=BIG_RED, fg="red", justify=tk.LEFT, anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=(4, 8))

        ttk.Separator(delta_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        tk.Label(
            delta_frame, textvariable=self.delta_info_var,
            font=INFO_FONT, fg="gray40", justify=tk.LEFT, anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=(4, 6))
        # ────────────────────────────────────────────────────────────────────────

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
        self._update_fit_display()
        self._update_overlap_display()
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

        if self.fits["left"] and self.fits["right"]:
            try:
                k    = int(self.anchor_k_var.get())
                m    = int(self.scale_ref_m_var.get())
                k_tw = int(self.anchor_k_tweezer_var.get())
                m_tw = int(self.scale_ref_m_tweezer_var.get())
            except ValueError:
                self.delta_values_var.set("Error: all four index fields must be integers.")
                self.delta_formula_var.set("")
                self.delta_info_var.set("")
                self.canvas.draw()
                return
            al   = compute_alignment(
                self.fits["left"], self.fits["right"], self.calibration_scale,
                k, m, k_tw, m_tw,
            )
            u    = al["units"]
            k    = al["anchor_k"]
            m    = al["scale_ref_m"]
            k_tw = al["anchor_k_tweezer"]
            m_tw = al["scale_ref_m_tweezer"]
            n    = al["n_pairs"]
            self.delta_values_var.set(
                f"  Δy'  =  {al['shift_y_rot']:+.6f}  {u}\n"
                f"\n"
                f"  Δx'  =  {al['shift_x_rot']:+.6f}  {u}\n"
                f"\n"
                f"  a    =  {al['scale_a']:+.6f}  {u} / tweezer"
            )
            self.delta_formula_var.set(
                f"  right_y'[t]  +  ({al['shift_y_rot']:+.6f})  +  (t \u2212 {k_tw})\u00b7({al['scale_a']:+.6f})  =  left_y'[t]"
            )
            self.delta_info_var.set(
                f"  {n} detected pairs  |  0 = topmost \u2192 {n-1} = bottommost  |  t = tweezer index\n"
                f"  negative y' = up, positive y' = down  |  positive a = expand right array\n"
                f"  Anchor k: pair {k}, tweezer {k_tw}   \u2022   Scale ref m: pair {m}, tweezer {m_tw}"
            )
        else:
            self.delta_values_var.set("")
            self.delta_formula_var.set("")
            self.delta_info_var.set("Fit both images to see δ results.")
        self.canvas.draw()

    def _update_fit_display(self) -> None:
        """Rebuild the Fits & Residuals tab with one row per fitted peak."""
        if self.fit_inner_frame is None or self.fit_tk_canvas is None:
            return

        for widget in self.fit_inner_frame.winfo_children():
            widget.destroy()
        self._fit_mpl_canvas = None

        left_fits  = sorted(self.fits["left"],  key=lambda f: f["row"])
        right_fits = sorted(self.fits["right"], key=lambda f: f["row"])
        all_items: list[tuple[str, int, dict]] = (
            [("Left",  i, f) for i, f in enumerate(left_fits)]
            + [("Right", i, f) for i, f in enumerate(right_fits)]
        )

        if not all_items:
            ttk.Label(
                self.fit_inner_frame,
                text="Run Fit on both images to see individual peak fits and residuals.",
                padding=20,
            ).pack()
            self.fit_tk_canvas.configure(scrollregion=self.fit_tk_canvas.bbox("all"))
            return

        try:
            patch_radius = int(self.param_vars["patch"].get())
        except ValueError:
            patch_radius = 12

        n_rows = len(all_items)
        row_height_in = 2.2
        # 4 columns: [coords | data | fit | residual]
        # coords column is narrower than the image columns
        fig = Figure(figsize=(9.5, n_rows * row_height_in))
        gs = fig.add_gridspec(
            n_rows, 4,
            width_ratios=[1.1, 2, 2, 2],
            hspace=0.6, wspace=0.25,
            left=0.02, right=0.98, top=0.98, bottom=0.02,
        )

        for row_idx, (side, peak_idx, fit) in enumerate(all_items):
            image = self.images[side.lower()]
            if image is None:
                continue

            # Extract patch around fitted centre
            pr = int(round(fit["row"]))
            pc = int(round(fit["col"]))
            y0 = max(pr - patch_radius, 0)
            y1 = min(pr + patch_radius + 1, image.shape[0])
            x0 = max(pc - patch_radius, 0)
            x1 = min(pc + patch_radius + 1, image.shape[1])
            patch = image[y0:y1, x0:x1].astype(float)

            # Reconstruct the fitted Gaussian over the same patch region
            y_grid, x_grid = np.mgrid[y0:y1, x0:x1]
            coords = np.vstack((x_grid.ravel(), y_grid.ravel()))
            params = np.array([
                fit["amplitude"], fit["col"], fit["row"],
                fit["sigma_x"], fit["sigma_y"],
                np.radians(fit["theta_deg"]), fit["offset"],
            ])
            fitted = elliptical_gaussian(params, coords).reshape(patch.shape)
            residual = patch - fitted

            vmin, vmax = patch.min(), patch.max()
            res_lim = max(float(np.abs(residual).max()), 1e-6)
            rms = float(np.sqrt(np.mean(residual ** 2)))

            ax_coord = fig.add_subplot(gs[row_idx, 0])
            ax_data  = fig.add_subplot(gs[row_idx, 1])
            ax_fit   = fig.add_subplot(gs[row_idx, 2])
            ax_res   = fig.add_subplot(gs[row_idx, 3])

            # ── Coordinate display ───────────────────────────────────────
            ax_coord.set_axis_off()
            ax_coord.text(
                0.5, 0.65,
                f"{side} #{peak_idx}",
                ha="center", va="center",
                fontsize=8, fontstyle="italic",
                transform=ax_coord.transAxes,
                color="#444444",
            )
            ax_coord.text(
                0.5, 0.38,
                f"x = {fit['col']:.2f}\ny = {fit['row']:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                transform=ax_coord.transAxes,
                linespacing=1.6,
            )

            # ── Data image with crosshair at fitted centre ───────────────
            ax_data.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
            cx = fit["col"] - x0  # centre in patch pixel coords
            cy = fit["row"] - y0
            ax_data.plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
            ax_data.set_title("data", fontsize=7)

            # ── Fitted Gaussian ──────────────────────────────────────────
            ax_fit.imshow(fitted, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
            ax_fit.set_title(
                f"fit  σx={fit['sigma_x']:.1f} σy={fit['sigma_y']:.1f} θ={fit['theta_deg']:.1f}°",
                fontsize=7,
            )

            # ── Residual ─────────────────────────────────────────────────
            ax_res.imshow(residual, cmap="RdBu_r", vmin=-res_lim, vmax=res_lim, origin="upper")
            ax_res.set_title(f"residual  rms={rms:.1f}", fontsize=7)

            for ax in (ax_data, ax_fit, ax_res):
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)

        self._fit_mpl_canvas = FigureCanvasTkAgg(fig, master=self.fit_inner_frame)
        self._fit_mpl_canvas.draw()
        self._fit_mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fit_inner_frame.update_idletasks()
        self.fit_tk_canvas.configure(scrollregion=self.fit_tk_canvas.bbox("all"))

    def _update_overlap_display(self) -> None:
        """Rebuild the Overlap tab: one zoomed row per matched peak pair."""
        if self.overlap_inner_frame is None or self.overlap_tk_canvas is None:
            return

        for widget in self.overlap_inner_frame.winfo_children():
            widget.destroy()
        self._overlap_mpl_canvas = None

        left_img  = self.images["left"]
        right_img = self.images["right"]

        if left_img is None or right_img is None:
            ttk.Label(
                self.overlap_inner_frame,
                text="Load and fit both images to see the overlap.",
                padding=20,
            ).pack()
            self.overlap_tk_canvas.configure(
                scrollregion=self.overlap_tk_canvas.bbox("all")
            )
            return

        left_sorted  = sorted(self.fits["left"],  key=lambda f: f["row"])
        right_sorted = sorted(self.fits["right"], key=lambda f: f["row"])
        pairs = list(zip(left_sorted, right_sorted))

        if not pairs:
            ttk.Label(
                self.overlap_inner_frame,
                text="Fit both images first.",
                padding=20,
            ).pack()
            self.overlap_tk_canvas.configure(
                scrollregion=self.overlap_tk_canvas.bbox("all")
            )
            return

        try:
            patch_radius = int(self.param_vars["patch"].get())
        except ValueError:
            patch_radius = 12

        def norm_patch(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(float)
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn) if mx > mn + 1e-9 else np.zeros_like(a)

        n_rows = len(pairs)
        row_height_in = 4.5
        fig = Figure(figsize=(13.0, n_rows * row_height_in))
        gs = fig.add_gridspec(
            n_rows, 4,
            width_ratios=[0.7, 2, 2, 2],
            hspace=0.35, wspace=0.15,
            left=0.02, right=0.98, top=0.98, bottom=0.02,
        )

        h_l, w_l = left_img.shape[:2]
        h_r, w_r = right_img.shape[:2]

        for row_idx, (lf, rf) in enumerate(pairs):
            # Crop region anchored on the LEFT peak centre
            pr = int(round(lf["row"]))
            pc = int(round(lf["col"]))
            y0 = max(pr - patch_radius, 0)
            y1 = min(pr + patch_radius + 1, h_l)
            x0 = max(pc - patch_radius, 0)
            x1 = min(pc + patch_radius + 1, w_l)

            left_patch = norm_patch(left_img[y0:y1, x0:x1])

            # Extract the same pixel region from the right image (clipped to its bounds)
            ry0 = max(y0, 0);  ry1 = min(y1, h_r)
            rx0 = max(x0, 0);  rx1 = min(x1, w_r)
            right_raw = np.zeros_like(left_patch)
            if ry1 > ry0 and rx1 > rx0:
                rp = norm_patch(right_img[ry0:ry1, rx0:rx1])
                right_raw[:rp.shape[0], :rp.shape[1]] = rp
            right_patch = right_raw

            diff = left_patch - right_patch

            # Sub-pixel offsets between fitted centres
            dx = rf["col"] - lf["col"]
            dy = rf["row"] - lf["row"]

            # Marker positions in patch pixel coordinates
            lx = lf["col"] - x0;  ly = lf["row"] - y0   # left peak (always centre)
            rx = rf["col"] - x0;  ry = rf["row"] - y0   # right peak (may be offset)

            # ── Coordinate info panel ────────────────────────────────────
            ax_info = fig.add_subplot(gs[row_idx, 0])
            ax_info.set_axis_off()
            ax_info.text(0.5, 0.72, f"Pair {row_idx}",
                         ha="center", va="center", fontsize=8, fontstyle="italic",
                         color="#444444", transform=ax_info.transAxes)
            ax_info.text(0.5, 0.42,
                         f"Δx = {dx:+.2f} px\nΔy = {dy:+.2f} px",
                         ha="center", va="center", fontsize=11, fontweight="bold",
                         linespacing=1.7, transform=ax_info.transAxes)

            # ── Image panels ─────────────────────────────────────────────
            ax_l = fig.add_subplot(gs[row_idx, 1])
            ax_r = fig.add_subplot(gs[row_idx, 2])
            ax_d = fig.add_subplot(gs[row_idx, 3])

            ax_l.imshow(left_patch,  cmap="gray", vmin=0, vmax=1, origin="upper")
            ax_r.imshow(right_patch, cmap="gray", vmin=0, vmax=1, origin="upper")
            lim = max(float(np.abs(diff).max()), 1e-9)
            ax_d.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")

            # Mark both peak centres on every panel
            marker_kw_l = dict(color="cyan",   marker="+", markersize=12,
                               markeredgewidth=1.5, linestyle="none")
            marker_kw_r = dict(color="orange", marker="x", markersize=10,
                               markeredgewidth=1.5, linestyle="none")
            for ax in (ax_l, ax_r, ax_d):
                ax.plot(lx, ly, **marker_kw_l)
                ax.plot(rx, ry, **marker_kw_r)

            ax_l.set_title("left",            fontsize=7)
            ax_r.set_title("right (same coords)", fontsize=7)
            ax_d.set_title("left − right",    fontsize=7)

            for ax in (ax_l, ax_r, ax_d):
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)

        self._overlap_mpl_canvas = FigureCanvasTkAgg(fig, master=self.overlap_inner_frame)
        self._overlap_mpl_canvas.draw()
        self._overlap_mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.overlap_inner_frame.update_idletasks()
        self.overlap_tk_canvas.configure(
            scrollregion=self.overlap_tk_canvas.bbox("all")
        )

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

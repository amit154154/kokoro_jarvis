#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Kokoro mixture training with a stylish circular mixer:
- Moving point shows convex-combo of anchors (voices)
- Per-voice bars with names & percentages
- Play/Pause, scrub, speed control
- Export MP4 video (includes the bars), fast by default

Usage:
  # from W&B
  python kokoro_coeff_viz.py --run_path ENTITY/PROJECT/RUN_ID

  # from local CSV (exported from W&B)
  python kokoro_coeff_viz.py --csv history.csv

  # video export only (no GUI play needed)
  python kokoro_coeff_viz.py --csv history.csv --export_mp4 out.mp4
"""

import argparse
import math
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time

# Optional deps
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    import imageio
    _HAS_IMGIO = True
except Exception:
    _HAS_IMGIO = False

try:
    from PIL import ImageGrab
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# ----------------- Data loading -----------------
def load_history_from_wandb(run_path: str):
    if not _HAS_WANDB:
        raise RuntimeError("wandb is not installed. pip install wandb")
    api = wandb.Api()
    run = api.run(run_path)
    df = run.history(pandas=True)
    if df is None or df.empty:
        raise RuntimeError("No history found for this run.")
    coef_cols = [c for c in df.columns if c.startswith("coef/")]
    if not coef_cols:
        raise RuntimeError("No 'coef/<voice>' columns in this run.")
    if "trainer/global_step" in df and df["trainer/global_step"].notna().any():
        steps = df["trainer/global_step"].to_numpy(dtype=float)
    elif "_step" in df:
        steps = df["_step"].to_numpy(dtype=float)
    else:
        steps = np.arange(len(df), dtype=float)
    voices = [c.split("/", 1)[1] for c in coef_cols]
    coefs = df[coef_cols].to_numpy(dtype=float)
    coefs = np.nan_to_num(coefs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = coefs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    coefs = coefs / row_sums
    return steps, voices, coefs


def load_history_from_csv(csv_path: str):
    import csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise RuntimeError("CSV seems empty.")
    header = rows[0].keys()
    coef_cols = [c for c in header if c.startswith("coef/")]
    if not coef_cols:
        raise RuntimeError("CSV does not contain any 'coef/<voice>' columns.")
    if "trainer/global_step" in header:
        step_key = "trainer/global_step"
    elif "_step" in header:
        step_key = "_step"
    else:
        step_key = None

    steps, coefs_list = [], []
    for i, r in enumerate(rows):
        step = float(r[step_key]) if step_key and r.get(step_key, "") != "" else float(i)
        steps.append(step)
        row = []
        for c in coef_cols:
            v = r.get(c, "")
            try:
                row.append(float(v))
            except Exception:
                row.append(0.0)
        coefs_list.append(row)

    steps = np.array(steps, dtype=float)
    coefs = np.array(coefs_list, dtype=float)
    coefs = np.nan_to_num(coefs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = coefs.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    coefs = coefs / row_sums
    voices = [c.split("/", 1)[1] for c in coef_cols]
    return steps, voices, coefs


# ----------------- Geometry helpers -----------------
def compute_anchor_positions(n, center, radius):
    cx, cy = center
    pos = []
    for i in range(n):
        theta = -math.pi / 2 + i * (2 * math.pi / n)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pos.append((x, y))
    return pos


def point_from_weights(weights, anchors, center, radius):
    xs = np.array([ax for (ax, _) in anchors], dtype=np.float32)
    ys = np.array([ay for (_, ay) in anchors], dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if w.sum() <= 0:
        w = np.full_like(w, 1.0 / len(w))
    w = w / w.sum()
    wx, wy = float((w * xs).sum()), float((w * ys).sum())
    cx, cy = center
    dx, dy = wx - cx, wy - cy
    r = math.hypot(dx, dy)
    if r <= radius or r == 0:
        return wx, wy
    s = radius / r
    return cx + dx * s, cy + dy * s


# ----------------- Palette -----------------
PALETTE = [
    "#4cc9f0", "#f72585", "#7209b7", "#3a0ca3",
    "#4895ef", "#4361ee", "#2ec4b6", "#ff9f1c",
    "#ef476f", "#06d6a0", "#ffd166", "#118ab2",
]


# ----------------- GUI -----------------
class CoeffVizGUI:
    def __init__(self, master, steps, voices, weights, radius=180, video_speed=180):
        self.master = master
        self.master.title("Kokoro Coefficient Visualizer (Circular + Bars)")

        self.steps = steps
        self.voices = voices
        self.weights = weights
        self.T, self.N = weights.shape

        self.radius = radius
        self.center = (radius + 40, radius + 40)

        # state
        self.playing = False
        self.idx = 0
        self.trail_enabled = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=24.0)
        self.video_speed_default = max(60.0, float(video_speed))

        # smoothing state
        first_row = weights[0] if self.T > 0 else np.ones((len(voices),), dtype=float) / max(1, len(voices))
        self._last_valid_w = first_row / max(1e-8, float(np.nansum(first_row)))
        self._ema_w = self._last_valid_w.copy()
        self._ema_alpha = 0.2

        # layout frames
        self.left_frame = ttk.Frame(self.master)
        self.left_frame.grid(row=0, column=0, rowspan=10, padx=10, pady=10)
        self.right_frame = ttk.Frame(self.master)
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 10), pady=10)

        # circle canvas
        self.canvas_w = int(self.center[0] * 2)
        self.canvas_h = int(self.center[1] * 2)
        self.canvas = tk.Canvas(self.left_frame, width=self.canvas_w, height=self.canvas_h, bg="#111")
        self.canvas.grid(row=0, column=0)
        self.anchors = compute_anchor_positions(self.N, self.center, self.radius)
        self._draw_wheel()

        self.point_handle = None
        self.glow_handle = None
        self.trail_polyline = None
        self.trail_points = []

        # bars
        self.bars_w = self.canvas_w
        self.bars_h = 28 * self.N + 24
        self.bars = tk.Canvas(self.left_frame, width=self.bars_w, height=self.bars_h, bg="#0f0f0f", highlightthickness=0)
        self.bars.grid(row=1, column=0, pady=(8, 0))

        # controls
        ttk.Label(self.right_frame, text="Voices:").pack(anchor="w")
        ttk.Label(self.right_frame, text=", ".join(self.voices)).pack(anchor="w", pady=(0, 8))
        self.step_label = ttk.Label(self.right_frame, text="Step: —"); self.step_label.pack(anchor="w")
        self.weight_label = ttk.Label(self.right_frame, text="Weights: —"); self.weight_label.pack(anchor="w", pady=(0, 8))
        ttk.Label(self.right_frame, text="Position").pack(anchor="w")
        self._updating_scrub = False
        self.scrub = ttk.Scale(self.right_frame, from_=0, to=max(0, self.T - 1), orient="horizontal", command=self._on_scrub)
        self.scrub.pack(fill="x", pady=(0, 8))
        ttk.Label(self.right_frame, text="Speed (steps/s)").pack(anchor="w")
        self.speed_scale = ttk.Scale(self.right_frame, from_=1.0, to=120.0, variable=self.speed, orient="horizontal")
        self.speed_scale.pack(fill="x", pady=(0, 8))
        self.trail_chk = ttk.Checkbutton(self.right_frame, text="Show trail", variable=self.trail_enabled, command=self._redraw_trail)
        self.trail_chk.pack(anchor="w", pady=(0, 8))
        btns = ttk.Frame(self.right_frame); btns.pack(fill="x", pady=(8, 8))
        self.play_btn = ttk.Button(btns, text="Play", command=self._toggle_play)
        self.play_btn.pack(side="left", expand=True, fill="x")
        ttk.Button(btns, text="Pause", command=self._pause).pack(side="left", expand=True, fill="x", padx=(6, 0))
        ex_btns = ttk.Frame(self.right_frame); ex_btns.pack(fill="x", pady=(8, 8))
        ttk.Button(ex_btns, text="Export PNG…", command=self._export_png).pack(side="left", expand=True, fill="x")
        ttk.Button(ex_btns, text="Export MP4…", command=self._export_mp4).pack(side="left", expand=True, fill="x", padx=(6, 0))

        self._go_to_index(0)
        self._tick_loop()

    # helpers
    def _safe_normalize(self, w):
        w = np.asarray(w, dtype=float)
        s = float(np.nansum(w))
        if (not np.isfinite(s)) or s <= 1e-8:
            return self._last_valid_w
        return w / s

    # drawing
    def _draw_wheel(self):
        cx, cy = self.center; R = self.radius
        self.canvas.create_oval(cx - R, cy - R, cx + R, cy + R, outline="#666", width=2)
        for (x, y), name in zip(self.anchors, self.voices):
            color = PALETTE[hash(name) % len(PALETTE)]
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")
            self.canvas.create_line(cx, cy, x, y, fill="#222")
            dx, dy = x - cx, y - cy
            ln = math.hypot(dx, dy) or 1.0
            lx, ly = x + 12 * (dx / ln), y + 12 * (dy / ln)
            self.canvas.create_text(lx, ly, text=name, fill="#ddd", anchor="center", font=("TkDefaultFont", 9, "bold"))

    def _draw_point(self, x, y):
        if self.point_handle is not None: self.canvas.delete(self.point_handle)
        if self.glow_handle is not None: self.canvas.delete(self.glow_handle)
        self.glow_handle = self.canvas.create_oval(x - 14, y - 14, x + 14, y + 14, outline="", fill="#ddd")
        self.point_handle = self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#fb5", outline="#000")

    def _redraw_trail(self):
        if self.trail_polyline is not None: self.canvas.delete(self.trail_polyline); self.trail_polyline = None
        if self.trail_enabled.get() and len(self.trail_points) >= 2:
            flat = [c for xy in self.trail_points for c in xy]
            self.trail_polyline = self.canvas.create_line(*flat, fill="#888")

    def _draw_bars(self, weights_row):
        self.bars.delete("all")
        margin, name_w = 10, 140
        bar_w = self.bars_w - (margin * 2 + name_w + 60)
        y = 12
        for i, name in enumerate(self.voices):
            w = float(max(0.0, min(1.0, weights_row[i])))
            color = PALETTE[hash(name) % len(PALETTE)]
            self.bars.create_text(margin, y + 10, text=name, anchor="w", fill="#ddd", font=("TkDefaultFont", 10, "bold"))
            x0 = margin + name_w
            self.bars.create_rectangle(x0, y, x0 + bar_w, y + 20, fill="#1a1a1a", outline="#333")
            self.bars.create_rectangle(x0, y, x0 + int(bar_w * w), y + 20, fill=color, outline=color)
            pct = f"{w * 100:.1f}%"
            self.bars.create_text(x0 + bar_w + 8, y + 10, text=pct, anchor="w", fill="#ccc")
            y += 28

    # update
    def _go_to_index(self, i, push_scrub=True):
        i = max(0, min(self.T - 1, int(i)))
        self.idx = i
        raw = self.weights[i]
        w_now = self._safe_normalize(raw)
        self._last_valid_w = w_now
        self._ema_w = (1 - self._ema_alpha) * self._ema_w + self._ema_alpha * w_now
        w_plot = self._safe_normalize(self._ema_w)
        x, y = point_from_weights(w_plot, self.anchors, self.center, self.radius)
        self._draw_point(x, y)
        step = self.steps[i] if i < len(self.steps) else i
        self.step_label.config(text=f"Step: {int(step)}  (idx {i+1}/{self.T})")
        pieces = [f"{name}:{w_plot[j]:.2f}" for j, name in enumerate(self.voices)]
        self.weight_label.config(text="Weights: " + "  ".join(pieces))
        if self.trail_points:
            px, py = self.trail_points[-1]; dist = math.hypot(x - px, y - py)
            if np.isfinite(dist) and dist <= self.radius * 1.25:
                self.trail_points.append((x, y))
        else:
            self.trail_points.append((x, y))
        if len(self.trail_points) > 2000: self.trail_points = self.trail_points[-2000:]
        self._redraw_trail()
        self._draw_bars(w_plot)
        if push_scrub:
            try:
                self._updating_scrub = True; self.scrub.set(i)
            finally:
                self._updating_scrub = False

    # controls
    def _on_scrub(self, _val):
        if self._updating_scrub: return
        try: idx = int(float(_val))
        except Exception: return
        self._go_to_index(idx, push_scrub=False)

    def _toggle_play(self): self.playing = not self.playing; self.play_btn.config(text=("Pause" if self.playing else "Play"))
    def _pause(self): self.playing = False; self.play_btn.config(text="Play")

    def _tick_loop(self):
        self.master.after(16, self._tick_loop)
        if not self.playing or self.T == 0: return
        spd = max(1.0, float(self.speed.get())); inc = spd / 60.0
        new_idx = self.idx + inc
        if new_idx >= self.T - 1: new_idx = self.T - 1; self._pause()
        self._go_to_index(int(new_idx))

    # export
    def _bbox_stage(self):
        x0 = self.master.winfo_rootx() + self.left_frame.winfo_x()
        y0 = self.master.winfo_rooty() + self.left_frame.winfo_y()
        w = max(self.canvas_w, self.bars_w); h = self.canvas_h + 8 + self.bars_h
        return tuple(map(int, (x0, y0, x0 + w, y0 + h)))
    def _export_png(self):
        if not _HAS_PIL: messagebox.showerror("Export error", "Pillow not installed."); return
        bbox = self._bbox_stage(); img = ImageGrab.grab(bbox=bbox)
        f = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="coeff_viz.png")
        if f: img.save(f); messagebox.showinfo("Saved", f"Saved: {f}")

    def _export_mp4(self):
        if not _HAS_IMGIO or not _HAS_PIL:
            messagebox.showerror("Export error", "Install imageio and pillow."); return
        f = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")], initialfile="coeff_viz.mp4")
        if not f: return
        steps_per_sec, fps = self.video_speed_default, 60
        was_playing, old_idx = self.playing, self.idx; self._pause(); bbox = self._bbox_stage()
        try:
            with imageio.get_writer(
                    f, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8
            ) as wtr:
                # fixed increment so we always advance
                inc = max(1, int(round(steps_per_sec / fps)))

                for i in range(0, self.T, inc):
                    # draw the next frame
                    self._go_to_index(i)
                    # force Tk to repaint *now* so ImageGrab sees the updated pixels
                    self.master.update_idletasks()
                    self.master.update()

                    # grab and write
                    img = ImageGrab.grab(bbox=bbox)
                    frame = np.asarray(img.convert("RGB"))
                    wtr.append_data(frame)

                    # tiny sleep helps macOS avoid throttling the UI thread
                    time.sleep(0.002)

            messagebox.showinfo("Saved", f"Saved: {f}")
        finally:
            self._go_to_index(old_idx); self.playing = was_playing
            self.play_btn.config(text=("Pause" if self.playing else "Play"))


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Visualize Kokoro mixture coefficients.")
    ap.add_argument("--run_path", type=str, default="amit154154/kokoro_opt/nrtl7bn1", help="W&B path ENTITY/PROJECT/RUN_ID")
    ap.add_argument("--csv", type=str, default="/Users/mac/PycharmProjects/Jarvis_Phone/scripts/wandb_run.csv", help="Local CSV exported from W&B history")
    ap.add_argument("--radius", type=int, default=180)
    ap.add_argument("--export_mp4", type=str, default="")
    ap.add_argument("--video_speed", type=float, default=180.0)
    args = ap.parse_args()

    if not args.run_path and not args.csv:
        print("Provide either --run_path or --csv", file=sys.stderr); sys.exit(1)
    steps, voices, weights = (load_history_from_csv(args.csv) if args.csv else load_history_from_wandb(args.run_path))

    order = np.argsort(steps); steps, weights = steps[order], weights[order]
    keep = [0]
    for i in range(1, len(steps)):
        if steps[i] != steps[keep[-1]]: keep.append(i)
    steps, weights = steps[keep], weights[keep]

    # forward/back-fill invalid rows
    if len(weights) > 0:
        n = weights.shape[1]; first_valid = None
        for i in range(len(weights)):
            s = float(np.nansum(weights[i]))
            if np.isfinite(s) and s > 1e-8: first_valid = i; break
        if first_valid is None:
            weights[:] = 1.0 / max(1, n)
        else:
            weights[:first_valid] = weights[first_valid]
            last = weights[first_valid].copy()
            for i in range(first_valid, len(weights)):
                s = float(np.nansum(weights[i]))
                if (not np.isfinite(s)) or s <= 1e-8: weights[i] = last
                else: last = weights[i]

    root = tk.Tk()
    gui = CoeffVizGUI(root, steps=steps, voices=voices, weights=weights, radius=args.radius, video_speed=args.video_speed)

    if args.export_mp4:
        root.update_idletasks(); root.update()
        if not _HAS_IMGIO or not _HAS_PIL: print("Install imageio, pillow", file=sys.stderr); sys.exit(2)
        f = args.export_mp4; steps_per_sec, fps = max(60.0, float(args.video_speed)), 60
        was_playing, old_idx = gui.playing, gui.idx; gui._pause(); bbox = gui._bbox_stage()
        with imageio.get_writer(
                f, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8
        ) as wtr:
            inc = max(1, int(round(steps_per_sec / fps)))
            bbox = gui._bbox_stage()
            bbox = tuple(map(int, bbox))

            for i in range(0, gui.T, inc):
                gui._go_to_index(i)
                # force paint before grabbing
                root.update_idletasks()
                root.update()

                img = ImageGrab.grab(bbox=bbox)
                frame = np.asarray(img.convert("RGB"))
                wtr.append_data(frame)
                time.sleep(0.002)
        gui._go_to_index(old_idx); gui.playing = was_playing; gui.play_btn.config(text=("Pause" if gui.playing else "Play"))
        print(f"[saved] {f}"); root.destroy(); return

    root.mainloop()


if __name__ == "__main__":
    main()
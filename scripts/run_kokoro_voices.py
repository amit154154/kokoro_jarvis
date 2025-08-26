
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kokoro Voice Mixer — Radial Wheel Edition
# - Cleaner, more informative visualization:
#   • Circular "voice wheel" with colored anchors
#   • Draggable mix handle with soft glow + snapping
#   • Live weight bars with percentages
#   • Fusion text apply still supported
# - Same functionality as before (text, speed, output path, synthesis)
# - No extra dependencies beyond numpy, tkinter, kokoro, soundfile, torch

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import threading, math
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

SR = 24000

DEFAULT_VOICES = [
    "af_sarah", "af_jessica", "am_adam", "af_alloy",
    "af_bella", "am_michael", "af_heart", "af_nicole",
]

# Soft color palette (distinct but friendly)
PALETTE = [
    "#6E9FFB", "#F28D85", "#8CD17D", "#FFD166",
    "#BFA0FF", "#6BD1D2", "#FF9F1C", "#FF6FB5",
]


class VoiceMixerGUI:
    def __init__(self, master, lang_code="a", voices=None, radius=180):
        self.master = master
        master.title("Kokoro Voice Mixer — Radial Wheel")

        self.lang_code = lang_code
        self.voices = voices or DEFAULT_VOICES
        self.colors = (PALETTE * ((len(self.voices) + len(PALETTE) - 1) // len(PALETTE)))[:len(self.voices)]
        self.radius = radius
        self.center = (radius + 50, radius + 50)  # a bit more padding

        # Pipeline + model
        self.pipe = KPipeline(lang_code=self.lang_code)
        self.device = self.pipe.model.device

        # Load voice packs
        self.packs = {}
        self._load_voices()

        # ----- UI -----
        self._build_ui()

        # Anchor positions around a circle
        self.anchor_positions = self._compute_anchor_positions(len(self.voices))
        self._draw_wheel()

        # Mixer point (draggable) starts at uniform weights (center)
        self.mix_pos = [self.center[0], self.center[1]]
        self.mix_handle = None
        self._draw_mixer_point()

        self.snap_radius = 18  # px
        self.generating = False
        self._update_weights()

        # Hover highlight state
        self._hover_anchor_idx = -1

    # ---------- Voice / math ----------
    def _load_voices(self):
        for v in self.voices:
            pack = self.pipe.load_voice(v)
            if not isinstance(pack, torch.FloatTensor):
                pack = torch.tensor(pack)
            self.packs[v] = pack.to(torch.float32)

    def _compute_anchor_positions(self, n: int) -> List[Tuple[float, float]]:
        cx, cy = self.center
        R = self.radius
        pos = []
        for i in range(n):
            theta = -math.pi/2 + i * (2 * math.pi / n)  # start at top, clockwise
            x = cx + R * math.cos(theta)
            y = cy + R * math.sin(theta)
            pos.append((x, y))
        return pos

    def _weights_from_point(self, x, y, power=2.0):
        """Inverse-distance weighting with SNAP to nearest anchor for easy selection."""
        # Snap check
        dmin = float("inf"); imin = -1
        for i, (ax, ay) in enumerate(self.anchor_positions):
            d = math.dist((x, y), (ax, ay))
            if d < dmin:
                dmin, imin = d, i
        if dmin <= self.snap_radius:
            w = np.zeros(len(self.anchor_positions), dtype=np.float32)
            w[imin] = 1.0
            return w, imin

        dists = np.array([max(1e-6, math.dist((x, y), (ax, ay))) for (ax, ay) in self.anchor_positions],
                         dtype=np.float32)
        inv = 1.0 / (dists ** power)
        w = inv / inv.sum()
        return w, -1

    def _blend_packs(self, weights):
        shapes = {tuple(self.packs[v].shape) for v in self.voices}
        if len(shapes) != 1:
            raise RuntimeError(f"Voice packs have differing shapes: {shapes}")
        stacked = torch.stack([self.packs[v] for v in self.voices], dim=0)  # [N, ...]
        w = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim()-1)))
        mixed = (w * stacked).sum(dim=0)
        return mixed

    def _point_from_weights(self, weights: np.ndarray):
        xs = np.array([ax for (ax, _) in self.anchor_positions], dtype=np.float32)
        ys = np.array([ay for (_, ay) in self.anchor_positions], dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)
        if w.sum() <= 0: w = np.full_like(w, 1.0 / len(w))
        w = w / w.sum()
        wx = float((w * xs).sum()); wy = float((w * ys).sum())
        return wx, wy

    # ---------- Synthesis ----------
    def _generate_audio(self):
        text = self.text_entry.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Missing text", "Please enter text to synthesize.")
            return
        speed = float(self.speed_var.get())
        outpath = self.out_path_var.get().strip()
        if not outpath:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = f"kokoro_mix_{ts}.wav"
            self.out_path_var.set(outpath)

        w, _ = self._weights_from_point(self.mix_pos[0], self.mix_pos[1])
        mixed_pack = self._blend_packs(w).to(self.device)

        def worker():
            try:
                self.generate_btn.config(state="disabled")
                self.generating = True

                waves = []
                for res in self.pipe(text, voice=mixed_pack, speed=speed, split_pattern=r"\n+"):
                    if res.audio is None:
                        continue
                    waves.append(res.audio.detach().float().cpu().numpy())
                if not waves:
                    raise RuntimeError("No audio returned from pipeline.")
                audio = np.concatenate(waves, axis=-1).astype("float32")
                sf.write(outpath, audio, SR)
                messagebox.showinfo("Saved", f"Saved: {outpath}")
            except Exception as e:
                messagebox.showerror("Generation error", str(e))
            finally:
                self.generating = False
                self.generate_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    # ---------- UI ----------
    def _build_ui(self):
        # Left: visual canvas
        self.canvas = tk.Canvas(self.master, width=self.center[0]*2, height=self.center[1]*2, bg="#0E1116", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=10, padx=10, pady=10)

        # Right: controls
        panel = ttk.Frame(self.master)
        panel.grid(row=0, column=1, sticky="nswe", padx=(0,10), pady=10)

        # Text input
        ttk.Label(panel, text="Text").pack(anchor="w")
        self.text_entry = tk.Text(panel, width=46, height=6)
        self.text_entry.pack(fill="x", pady=(2,10))
        self.text_entry.insert("1.0", "Hello! Drag the dot to mix the voices, then click Generate.")

        # Speed
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_frame = ttk.Frame(panel); speed_frame.pack(fill="x", pady=(0,8))
        ttk.Label(speed_frame, text="Speed").pack(side="left")
        self.speed_scale = ttk.Scale(speed_frame, from_=0.6, to=1.4, variable=self.speed_var, orient="horizontal")
        self.speed_scale.pack(side="left", fill="x", expand=True, padx=(8,0))

        # Output path
        ttk.Label(panel, text="Output WAV").pack(anchor="w")
        pfrm = ttk.Frame(panel); pfrm.pack(fill="x", pady=(0,8))
        self.out_path_var = tk.StringVar(value="")
        self.out_entry = ttk.Entry(pfrm, textvariable=self.out_path_var)
        self.out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(pfrm, text="Browse…", command=self._browse_out).pack(side="left", padx=(6,0))

        # Generate button
        self.generate_btn = ttk.Button(panel, text="Generate", command=self._generate_audio)
        self.generate_btn.pack(fill="x", pady=(6,10))

        # Weight bars header
        ttk.Label(panel, text="Mixture Weights").pack(anchor="w")
        self.bars_canvas = tk.Canvas(panel, height=14 * len(self.voices) + 10, bg="#0E1116", highlightthickness=0)
        self.bars_canvas.pack(fill="x", pady=(4,10))
        self._init_bars()

        # Fusion text
        ttk.Label(panel, text="Fusion weights (comma or name:value)").pack(anchor="w")
        fusion_frame = ttk.Frame(panel); fusion_frame.pack(fill="x", pady=(2,0))
        default_w = ", ".join([f"{1.0/len(self.voices):.2f}"] * len(self.voices))
        self.fusion_var = tk.StringVar(value=default_w)
        self.fusion_entry = ttk.Entry(fusion_frame, textvariable=self.fusion_var)
        self.fusion_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(fusion_frame, text="Apply", command=self._apply_fusion_text).pack(side="left", padx=(6,0))

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

    def _browse_out(self):
        f = filedialog.asksaveasfilename(defaultextension=".wav",
                                         filetypes=[("WAV", "*.wav")],
                                         initialfile=f"kokoro_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        if f:
            self.out_path_var.set(f)

    # ---------- Drawing the radial wheel ----------
    def _draw_wheel(self):
        cx, cy = self.center
        R = self.radius

        # radial background (soft rings)
        for i in range(8, 0, -1):
            r = int(R * i / 8)
            shade = 20 + i * 8
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="", fill=color)

        # outer ring
        self.canvas.create_oval(cx - R, cy - R, cx + R, cy + R, outline="#93A1B0", width=2)

        # ticks
        for i in range(24):
            a = i * (2 * math.pi / 24.0)
            x0 = cx + (R - 8) * math.cos(a); y0 = cy + (R - 8) * math.sin(a)
            x1 = cx + R * math.cos(a);       y1 = cy + R * math.sin(a)
            self.canvas.create_line(x0, y0, x1, y1, fill="#222")

        # anchors + labels
        self.anchor_handles = []
        for (x, y), name, col in zip(self.anchor_positions, self.voices, self.colors):
            # arc wedge hint
            self.canvas.create_oval(x - 11, y - 11, x + 11, y + 11, outline="", fill=col)
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#000", fill="#FFF")
            # name slightly outside circle
            dx, dy = x - cx, y - cy
            nx = x + 14 * (dx / (abs(dx) + abs(dy) + 1e-6))
            ny = y + 14 * (dy / (abs(dx) + abs(dy) + 1e-6))
            self.canvas.create_text(nx, ny, text=name, fill=col, anchor="center",
                                    font=("TkDefaultFont", 9, "bold"))
            self.anchor_handles.append((x, y))

    def _draw_mixer_point(self):
        x, y = self.mix_pos
        if hasattr(self, "mix_handle") and self.mix_handle is not None:
            self.canvas.delete(self.mix_handle)
            self.canvas.delete(self.mix_glow)
        # soft "glow" behind (light gray halo)
        self.mix_glow = self.canvas.create_oval(
            x - 14, y - 14, x + 14, y + 14,
            outline="", fill="#ddd"
        )
        # main handle (bright yellow)
        self.mix_handle = self.canvas.create_oval(
            x - 7, y - 7, x + 7, y + 7,
            fill="#fb5", outline="#000", width=1
        )
    def _clamp_to_circle(self, x, y):
        cx, cy = self.center
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)
        if r <= self.radius or r == 0:
            return x, y
        s = self.radius / r
        return cx + dx * s, cy + dy * s

    # ---------- Weight bars ----------
    def _init_bars(self):
        self.bar_rows = []
        h = 14
        W = 300
        pad = 4
        for i, (name, col) in enumerate(zip(self.voices, self.colors)):
            y0 = 5 + i * h
            # label
            lbl = self.bars_canvas.create_text(4, y0 + 7, text=f"{name}", fill="#D7E3F4", anchor="w")
            # background bar
            bg = self.bars_canvas.create_rectangle(110, y0 + 2, 110 + W, y0 + 12, fill="#1B2430", outline="")
            # value bar
            fg = self.bars_canvas.create_rectangle(110, y0 + 2, 110, y0 + 12, fill=col, outline="")
            # percentage
            pct = self.bars_canvas.create_text(110 + W + 6, y0 + 7, text="0%", fill="#A7B8CC", anchor="w")
            self.bar_rows.append((lbl, bg, fg, pct, W))

    def _update_bars(self, weights: np.ndarray):
        weights = np.asarray(weights, dtype=np.float32)
        for i, w in enumerate(weights):
            lbl, bg, fg, pct, W = self.bar_rows[i]
            w = float(max(0.0, min(1.0, w)))
            x1 = 110 + int(W * w)
            # resize fg rect
            self.bars_canvas.coords(fg, 110, self.bars_canvas.coords(fg)[1], x1, self.bars_canvas.coords(fg)[3])
            self.bars_canvas.itemconfig(pct, text=f"{w*100:.1f}%")

    # ---------- Events ----------
    def _on_canvas_click(self, e):
        x, y = self._clamp_to_circle(e.x, e.y)
        self.mix_pos = [x, y]
        self.canvas.delete("all")
        self._draw_wheel()
        self._draw_mixer_point()
        self._update_weights()

    def _on_canvas_drag(self, e):
        self._on_canvas_click(e)

    def _on_canvas_motion(self, e):
        # simple hover halo over nearest anchor
        x, y = e.x, e.y
        _, imin = self._weights_from_point(x, y)
        if imin != self._hover_anchor_idx:
            self._hover_anchor_idx = imin
            self.canvas.delete("all")
            self._draw_wheel()
            # draw hover halo
            if imin >= 0:
                ax, ay = self.anchor_positions[imin]
                self.canvas.create_oval(ax - 16, ay - 16, ax + 16, ay + 16, outline="#FFF", width=1, dash=(2, 2))
            self._draw_mixer_point()

    def _update_weights(self):
        w, _ = self._weights_from_point(self.mix_pos[0], self.mix_pos[1])
        self._update_bars(w)

        # update fusion box to reflect current weights (rounded)
        self.fusion_var.set(", ".join(f"{wi:.2f}" for wi in w))

    def _apply_fusion_text(self):
        raw = self.fusion_var.get().strip()
        if not raw:
            return
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        w = np.zeros(len(self.voices), dtype=np.float32)
        named = any(':' in p for p in parts)
        try:
            if named:
                lut = {name: i for i, name in enumerate(self.voices)}
                for p in parts:
                    k, v = p.split(':', 1)
                    k = k.strip(); v = float(v.strip())
                    if k not in lut:
                        raise ValueError(f"Unknown voice '{k}'. Valid: {', '.join(self.voices)}")
                    w[lut[k]] = max(0.0, v)
            else:
                nums = [float(x) for x in parts]
                if len(nums) != len(self.voices):
                    raise ValueError(f"Need {len(self.voices)} weights; got {len(nums)}")
                w = np.array([max(0.0, v) for v in nums], dtype=np.float32)
            if w.sum() <= 0:
                raise ValueError("All weights are zero.")
            w = w / w.sum()
        except Exception as e:
            messagebox.showerror("Weights parse error", str(e))
            return

        # Move point to convex combo of anchors
        x, y = self._point_from_weights(w)
        x, y = self._clamp_to_circle(x, y)
        self.mix_pos = [x, y]
        self.canvas.delete("all")
        self._draw_wheel()
        self._draw_mixer_point()
        self._update_weights()


def main():
    root = tk.Tk()
    # make ttk look a tad nicer
    try:
        root.call("tk", "scaling", 1.2)
    except Exception:
        pass
    app = VoiceMixerGUI(root, lang_code="a", voices=DEFAULT_VOICES, radius=180)
    root.mainloop()


if __name__ == "__main__":
    main()
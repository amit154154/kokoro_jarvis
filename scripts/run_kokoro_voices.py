# kokoro_voice_mixer_gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, math
from datetime import datetime

import numpy as np
import soundfile as sf



import torch
from kokoro import KPipeline

SR = 24000

DEFAULT_VOICES = [
    # You can change/extend this list. They must exist under the Kokoro repo `voices/`.
    "af_heart", "am_michael",
]

class VoiceMixerGUI:
    def __init__(self, master, lang_code="a", voices=None, radius=180):
        self.master = master
        master.title("Kokoro Voice Mixer")

        self.lang_code = lang_code
        self.voices = voices or DEFAULT_VOICES
        self.radius = radius
        self.center = (radius + 40, radius + 40)  # canvas padding

        # Pipeline + model
        self.pipe = KPipeline(lang_code=self.lang_code)
        self.device = self.pipe.model.device

        # Load voice packs (FloatTensor, usually [T, D])
        self.packs = {}
        self._load_voices()

        # ----- UI -----
        self._build_ui()

        # Place anchors around the circle
        self.anchor_positions = self._compute_anchor_positions(len(self.voices))
        self._draw_wheel()

        # Mixer point (draggable)
        self.mix_pos = [self.center[0], self.center[1]]  # start at center
        self.mix_handle = None
        self._draw_mixer_point()

        # Easier selection: snap if within N pixels
        self.snap_radius = 14  # px
        self.generating = False
        self._update_weights_label()

    # ---------- Voice / mix math ----------
    def _load_voices(self):
        for v in self.voices:
            pack = self.pipe.load_voice(v)
            if not isinstance(pack, torch.FloatTensor):
                pack = torch.tensor(pack)
            self.packs[v] = pack.to(torch.float32)

    def _compute_anchor_positions(self, n):
        cx, cy = self.center
        R = self.radius
        # Distribute equally; start at top (−90°)
        pos = []
        for i in range(n):
            theta = -math.pi/2 + i * (2*math.pi/n)
            x = cx + R * math.cos(theta)
            y = cy + R * math.sin(theta)
            pos.append((x, y))
        return pos

    def _weights_from_point(self, x, y, power=2.0):
        """Inverse-distance weighting with SNAP for easy selection."""
        # Snap to nearest anchor if close
        dmin = float("inf"); imin = -1
        for i, (ax, ay) in enumerate(self.anchor_positions):
            d = math.dist((x, y), (ax, ay))
            if d < dmin:
                dmin, imin = d, i
        if dmin <= self.snap_radius:
            w = np.zeros(len(self.anchor_positions), dtype=np.float32)
            w[imin] = 1.0
            return w
        # Otherwise inverse-distance weights (stable near center)
        dists = np.array([max(1e-6, math.dist((x, y), (ax, ay))) for (ax, ay) in self.anchor_positions],
                         dtype=np.float32)
        inv = 1.0 / (dists ** power)
        w = inv / inv.sum()
        return w

    def _blend_packs(self, weights):
        """Weighted average of voice packs (supports [T,D] or [D])."""
        shapes = {tuple(self.packs[v].shape) for v in self.voices}
        if len(shapes) != 1:
            raise RuntimeError(f"Voice packs have differing shapes: {shapes}")
        stacked = torch.stack([self.packs[v] for v in self.voices], dim=0)  # [N, ...]
        w = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim()-1)))
        mixed = (w * stacked).sum(dim=0)
        return mixed

    def _point_from_weights(self, weights):
        """Map a set of weights back to a point (convex combo of anchors)."""
        xs = np.array([ax for (ax, _) in self.anchor_positions], dtype=np.float32)
        ys = np.array([ay for (_, ay) in self.anchor_positions], dtype=np.float32)
        wx = float((weights * xs).sum())
        wy = float((weights * ys).sum())
        return self._clamp_to_circle(wx, wy)

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

        # Compute weights and mix pack
        w = self._weights_from_point(self.mix_pos[0], self.mix_pos[1])
        mixed_pack = self._blend_packs(w).to(self.device)

        # Synthesize in a worker thread
        def worker():
            try:
                self.generate_btn.config(state="disabled")
                self.generating = True

                # Run pipeline, concatenate all chunks
                waves = []
                for res in self.pipe(text, voice=mixed_pack, speed=speed, split_pattern=r"\n+"):
                    if res.audio is None:
                        continue
                    waves.append(res.audio.detach().float().cpu().numpy())

                if not waves:
                    raise RuntimeError("No audio returned from pipeline.")

                audio = np.concatenate(waves, axis=-1).astype("float32")
                sf.write(outpath, audio, SR)
                # Show confirmation dialog
                messagebox.showinfo("Saved", f"Saved: {outpath}")

            except Exception as e:
                messagebox.showerror("Generation error", str(e))
            finally:
                self.generating = False
                self.generate_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    # ---------- UI construction / events ----------
    def _build_ui(self):
        # Canvas (wheel)
        self.canvas = tk.Canvas(self.master, width=self.center[0]*2, height=self.center[1]*2, bg="#111")
        self.canvas.grid(row=0, column=0, rowspan=8, padx=10, pady=10)

        # Text input
        ttk.Label(self.master, text="Text").grid(row=0, column=1, sticky="w", padx=(0,6))
        self.text_entry = tk.Text(self.master, width=48, height=6)
        self.text_entry.grid(row=1, column=1, padx=(0,10), pady=(0,10))
        self.text_entry.insert("1.0", "Hello! Mix the voices by dragging the dot, then click Generate.")

        # Speed
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Label(self.master, text="Speed").grid(row=2, column=1, sticky="w")
        self.speed_scale = ttk.Scale(self.master, from_=0.6, to=1.4, variable=self.speed_var, orient="horizontal")
        self.speed_scale.grid(row=3, column=1, sticky="we", padx=(0,10))

        # Output path
        self.out_path_var = tk.StringVar(value="")
        ttk.Label(self.master, text="Output WAV").grid(row=4, column=1, sticky="w")
        pfrm = ttk.Frame(self.master)
        pfrm.grid(row=5, column=1, sticky="we", padx=(0,10), pady=(0,6))
        self.out_entry = ttk.Entry(pfrm, textvariable=self.out_path_var, width=40)
        self.out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(pfrm, text="Browse…", command=self._browse_out).pack(side="left", padx=(6,0))

        # Generate button
        self.generate_btn = ttk.Button(self.master, text="Generate", command=self._generate_audio)
        self.generate_btn.grid(row=6, column=1, sticky="we", padx=(0,10), pady=(6,10))

        # Weights readout
        self.weights_label = ttk.Label(self.master, text="Weights: —")
        self.weights_label.grid(row=7, column=1, sticky="w")

        # Manual weights input ("fusion")
        ttk.Label(self.master, text="Fusion weights (comma or name:value)").grid(row=8, column=1, sticky="w")
        fusion_frame = ttk.Frame(self.master)
        fusion_frame.grid(row=9, column=1, sticky="we", padx=(0,10))
        # default to uniform weights
        default_w = ", ".join([f"{1.0/len(self.voices):.2f}"] * len(self.voices))
        self.fusion_var = tk.StringVar(value=default_w)
        self.fusion_entry = ttk.Entry(fusion_frame, textvariable=self.fusion_var)
        self.fusion_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(fusion_frame, text="Apply", command=self._apply_fusion_text).pack(side="left", padx=(6,0))

        # Mouse bindings for the wheel
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

    def _browse_out(self):
        f = filedialog.asksaveasfilename(defaultextension=".wav",
                                         filetypes=[("WAV", "*.wav")],
                                         initialfile=f"kokoro_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        if f:
            self.out_path_var.set(f)

    def _draw_wheel(self):
        # Circle
        cx, cy = self.center
        R = self.radius
        self.canvas.create_oval(cx-R, cy-R, cx+R, cy+R, outline="#666", width=2)

        # Anchors
        for (x, y), name in zip(self.anchor_positions, self.voices):
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="#4cf", outline="")
            # Label (slight offset outward)
            dx = (x - cx); dy = (y - cy)
            lx = x + 12 * (dx / (abs(dx)+abs(dy)+1e-6))
            ly = y + 12 * (dy / (abs(dx)+abs(dy)+1e-6))
            self.canvas.create_text(lx, ly, text=name, fill="#ddd", anchor="center",
                                    font=("TkDefaultFont", 9, "bold"))

    def _draw_mixer_point(self):
        x, y = self.mix_pos
        if hasattr(self, "mix_handle") and self.mix_handle is not None:
            self.canvas.delete(self.mix_handle)
        self.mix_handle = self.canvas.create_oval(x-7, y-7, x+7, y+7, fill="#fb5", outline="#000")

    def _clamp_to_circle(self, x, y):
        cx, cy = self.center
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)
        R = self.radius
        if r <= R or r == 0:
            return x, y
        # project to circumference
        scale = R / r
        return cx + dx*scale, cy + dy*scale

    def _on_canvas_click(self, e):
        x, y = self._clamp_to_circle(e.x, e.y)
        self.mix_pos = [x, y]
        self._draw_mixer_point()
        self._update_weights_label()
        # sync fusion textbox with current weights
        w = self._weights_from_point(self.mix_pos[0], self.mix_pos[1])
        self.fusion_var.set(", ".join(f"{wi:.2f}" for wi in w))

    def _on_canvas_drag(self, e):
        self._on_canvas_click(e)

    def _update_weights_label(self):
        if not hasattr(self, 'weights_label'):
            return
        w = self._weights_from_point(self.mix_pos[0], self.mix_pos[1])
        pieces = [f"{name}:{w[i]:.2f}" for i, name in enumerate(self.voices)]
        self.weights_label.config(text="Weights: " + "  ".join(pieces))


    def _apply_fusion_text(self):
        """Parse weights text like '0.6,0.4' or 'af_heart:0.7, am_michael:0.3' and move the dot."""
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
        # Move the point to the convex combination of anchors
        x, y = self._point_from_weights(w)
        self.mix_pos = [x, y]
        self._draw_mixer_point()
        self._update_weights_label()


def main():
    root = tk.Tk()
    app = VoiceMixerGUI(root, lang_code="a", voices=DEFAULT_VOICES, radius=180)
    root.mainloop()

if __name__ == "__main__":
    main()
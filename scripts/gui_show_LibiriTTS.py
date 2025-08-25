#!/usr/bin/env python3
import sys, os, random, threading, subprocess, platform, re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- Audio ---------------- #
class AudioPlayer:
    def __init__(self):
        self.proc = None
        self.os = platform.system().lower()
        self._playing = False

    def is_playing(self):
        if self.os.startswith("win"):
            return self._playing
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.os.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
            self._playing = False
            return
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def play(self, wav_path: str):
        self.stop()
        if self.os == "darwin":
            self.proc = subprocess.Popen(["afplay", wav_path],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif self.os == "linux":
            import shutil
            player = shutil.which("aplay") or shutil.which("paplay")
            if not player:
                raise RuntimeError("Install 'aplay' or 'paplay' for playback.")
            self.proc = subprocess.Popen([player, wav_path],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif self.os.startswith("win"):
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._playing = True
        else:
            raise RuntimeError(f"Unsupported OS: {self.os}")

# ---------------- Indexing ---------------- #
DIGITS_ONLY = re.compile(r"^\d+$")

def load_speaker_desc(csv_path: Path) -> dict:
    """Load a mapping {speaker_id(str) -> description(str)} from a pipe-delimited csv.
    Each line format: <speaker_id>|<description>. Lines starting with '#' are ignored.
    """
    mapping = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('|', 1)
                if len(parts) != 2:
                    continue
                sid, desc = parts[0].strip(), parts[1].strip()
                if sid:
                    mapping[sid] = desc
    except Exception:
        # If file can't be read, return empty mapping silently
        return {}
    return mapping

def extract_speaker_id(root: Path, wav: Path) -> str:
    """
    Tries to get the speaker id as the first numeric directory under root.
    Falls back to the first component if needed.
    """
    rel = wav.relative_to(root)
    for part in rel.parts[:-1]:  # ignore filename
        if DIGITS_ONLY.match(part):
            return part
    return rel.parts[0] if rel.parts else "UNKNOWN"

def build_index(root_dir: Path):
    samples = []
    for wav in root_dir.rglob("*.wav"):
        norm_txt = wav.with_suffix(".normalized.txt")  # e.g., foo.wav -> foo.normalized.txt
        if not norm_txt.exists():
            continue
        spk = extract_speaker_id(root_dir, wav)
        samples.append((wav, norm_txt, spk))
    random.shuffle(samples)
    return samples

# ---------------- GUI ---------------- #
class BrowserGUI:
    def __init__(self, tk_root: tk.Tk, dataset_root: Path, spk_desc: dict | None = None):
        self.tk_root = tk_root
        self.dataset_root = dataset_root.resolve()
        self.samples = build_index(self.dataset_root)
        if not self.samples:
            messagebox.showerror("No Samples", f"No (wav, *.normalized.txt) pairs under:\n{self.dataset_root}")
            tk_root.destroy()
            return

        self.player = AudioPlayer()
        self.current = None
        self.spk_desc = spk_desc or {}

        tk_root.title("LibriTTS Random Browser")
        tk_root.geometry("1000x560")
        tk_root.minsize(800, 420)

        container = ttk.Frame(tk_root, padding=12); container.pack(fill="both", expand=True)

        top = ttk.Frame(container); top.pack(fill="x", pady=(0,10))
        ttk.Button(top, text="Next (N)", command=self.show_random).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Play / Stop (Space)", command=self.toggle_play).pack(side="left")

        info = ttk.Frame(container); info.pack(fill="x", pady=(0,8))
        lab_bold = ("Helvetica", 13, "bold")
        mono = ("SF Mono", 11)

        ttk.Label(info, text="Speaker ID:", font=lab_bold).grid(row=0, column=0, sticky="w", padx=(0,6))
        self.spk_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.spk_var).grid(row=0, column=1, sticky="w")

        ttk.Label(info, text="File:", font=lab_bold).grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.file_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.file_var, font=mono, wraplength=900, justify="left")\
            .grid(row=1, column=1, sticky="w", pady=(6,0))

        ttk.Label(info, text="Description:", font=lab_bold).grid(row=2, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.desc_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.desc_var, wraplength=900, justify="left")\
            .grid(row=2, column=1, sticky="w", pady=(6,0))

        ttk.Label(container, text="Normalized Text:", font=lab_bold).pack(anchor="w")
        self.text_box = tk.Text(container, height=12, wrap="word")
        self.text_box.configure(font=("Helvetica", 12))
        self.text_box.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value=f"Indexed {len(self.samples)} samples from {self.dataset_root}")
        ttk.Label(tk_root, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", side="bottom")

        tk_root.bind("<space>", lambda e: self.toggle_play())
        tk_root.bind("<KeyPress-n>", lambda e: self.show_random())
        tk_root.bind("<KeyPress-N>", lambda e: self.show_random())

        self.show_random()

    def show_random(self):
        if self.player.is_playing():
            self.player.stop()
        wav, txt, spk = random.choice(self.samples)
        self.current = (wav, txt, spk)

        self.spk_var.set(str(spk))
        self.file_var.set(str(wav))  # full path
        self.desc_var.set(self.spk_desc.get(str(spk), "—"))

        try:
            norm_text = Path(txt).read_text(encoding="utf-8").strip()
        except Exception as e:
            norm_text = f"[Error reading text: {e}]"

        self.text_box.config(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", norm_text or "—")
        self.text_box.config(state="disabled")

        rel = wav.relative_to(self.dataset_root)
        self.status_var.set(f"{rel}  |  Speaker {spk}")

    def toggle_play(self):
        if not self.current:
            return
        wav, _, _ = self.current
        try:
            if self.player.is_playing():
                self.player.stop()
            else:
                threading.Thread(target=self.player.play, args=(str(wav),), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

# ---------------- Main ---------------- #
def main():
    csv_path = Path("/Users/mac/PycharmProjects/Jarvis_Phone/data/LibriTTS_R/df1_en.csv")
    root_dir = Path("/Users/mac/PycharmProjects/Jarvis_Phone/data/LibriTTS_R/dev-clean")

    spk_map = {}
    if csv_path.is_file():
        spk_map = load_speaker_desc(csv_path)

    tk_root = tk.Tk()
    BrowserGUI(tk_root, root_dir, spk_map)
    tk_root.mainloop()

if __name__ == "__main__":
    main()
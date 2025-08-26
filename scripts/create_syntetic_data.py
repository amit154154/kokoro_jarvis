#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a synthetic speech dataset with Kokoro from a blended (mixed) voice.

Inputs:
  - Text file: one sentence/utterance per line
  - Output folder: WAVs are written here
  - Voice list + weights: either positional weights or "name:value" pairs

Example usages:

  # 1) Use positional weights with the default voice order
  #    (af_sarah, af_jessica, am_adam, af_alloy, af_bella, am_michael, af_heart, af_nicole)
  python make_kokoro_dataset.py \
    --text_file data/lines.txt \
    --out_dir out_kokoro \
    --weights 0.40,0.10,0.00,0.10,0.10,0.20,0.05,0.05

  # 2) Use explicit name:value pairs (order-independent)
  python make_kokoro_dataset.py \
    --text_file data/lines.txt \
    --out_dir out_kokoro \
    --voices af_sarah,af_jessica,am_adam,af_alloy \
    --weights af_sarah:0.6, am_adam:0.3, af_alloy:0.1

  # 3) Custom speed & language code
  python make_kokoro_dataset.py \
    --text_file data/lines.txt \
    --out_dir out_kokoro \
    --weights af_sarah:1.0 \
    --speed 1.05 \
    --lang a
"""

import argparse
import csv
import os
import re
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from tqdm import tqdm

# NOTE: The metadata now includes: filepath (relative to out_dir), transcript/text, duration_sec, has_speech.
# To use with KokoroJarvisSFTDataset:
#   ds = KokoroJarvisSFTDataset(audio_root=Path(OUT_DIR), metadata_csv=Path(OUT_DIR)/"metadata.csv", voice="af_heart")

SR = 24000

DEFAULT_VOICES = [
    "af_sarah", "af_jessica", "am_adam", "af_alloy",
    "af_bella", "am_michael", "af_heart", "af_nicole",
]
DEFAULT_COEFFS = [1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0]
DEFAULT_WEIGHTS_STR = "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
SPLIT_PATTERN = r"\n+"  # keep same behavior as your GUI; pipeline may stream chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize WAVs from blended Kokoro voice.")
    p.add_argument("--text_file", default="/Users/mac/PycharmProjects/Jarvis_Phone/data/syntetic_datasets/example_text.txt", help="Path to UTF-8 .txt file; one utterance per line.")
    p.add_argument("--out_dir", default="/Users/mac/PycharmProjects/Jarvis_Phone/data/syntetic_datasets/fully_sara", help="Output folder for WAV files.")
    p.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                   help="Comma-separated list of voice names (order for positional weights). "
                        f"Default: {','.join(DEFAULT_VOICES)}")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_STR,
                   help="Either comma-separated floats (positional, length must match --voices) OR name:value pairs separated by commas. "
                        "Examples: '0.5,0.5,0,0'  OR  'af_sarah:0.6, am_adam:0.4'")
    p.add_argument("--lang", default="a", help="Language code for Kokoro (default: 'a').")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.6–1.4 typically).")
    p.add_argument("--prefix", default="utt", help="Filename prefix for WAV files.")
    p.add_argument("--start_idx", type=int, default=1, help="Starting index for filename numbering.")
    p.add_argument("--metadata_csv", default="metadata.csv",
                   help="Name of metadata CSV written into out_dir (default: metadata.csv).")
    return p.parse_args()


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # Keep only non-empty lines
    return [ln for ln in lines if ln]


def parse_weights(raw: str, voices: List[str]) -> np.ndarray:
    """
    Accept either positional floats (len==len(voices)) or name:value pairs (order-independent).
    Returns normalized weights (sum==1). Also accepts list/tuple/ndarray of floats.
    """
    # If the caller provided a list/tuple/ndarray (e.g., via programmatic call), handle it directly.
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.array([max(0.0, float(x)) for x in raw], dtype=np.float32)
        if len(arr) != len(voices):
            raise ValueError(f"Positional weights length {len(arr)} != number of voices {len(voices)}")
        if arr.sum() <= 0:
            raise ValueError("All weights are zero.")
        return arr / arr.sum()

    if not isinstance(raw, str):
        raise TypeError(f"--weights must be a string or list-like; got {type(raw).__name__}")

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("No weights provided.")

    named = any(":" in p for p in parts)

    w = np.zeros(len(voices), dtype=np.float32)
    if named:
        lut = {name: i for i, name in enumerate(voices)}
        for p in parts:
            if ":" not in p:
                raise ValueError(f"Expected name:value pair but got '{p}'")
            k, v = p.split(":", 1)
            k = k.strip()
            try:
                val = float(v.strip())
            except Exception:
                raise ValueError(f"Could not parse weight for '{k}': '{v}'")
            if k not in lut:
                raise ValueError(f"Unknown voice '{k}'. Valid: {', '.join(voices)}")
            w[lut[k]] = max(0.0, val)
    else:
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                raise ValueError(f"Could not parse weight float: '{p}'")
        if len(nums) != len(voices):
            raise ValueError(f"Positional weights length {len(nums)} != number of voices {len(voices)}")
        w = np.array([max(0.0, v) for v in nums], dtype=np.float32)

    if w.sum() <= 0:
        raise ValueError("All weights are zero.")
    w = w / w.sum()
    return w


def safe_slug(s: str, max_len: int = 60) -> str:
    """Create a filename-friendly snippet from text."""
    s = s.strip()
    # take first sentence-ish or up to max_len
    s = re.split(r"[.?!]\s", s, maxsplit=1)[0]
    s = s[:max_len]
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-]", "", s)
    return s or "utt"


def load_and_blend_voices(pipe: KPipeline, voices: List[str], weights: np.ndarray, device: torch.device) -> torch.Tensor:
    """Load each voice pack, ensure same shape/dtype, and compute weighted blend."""
    packs = []
    shapes = set()
    for v in voices:
        pack = pipe.load_voice(v)
        if not isinstance(pack, torch.FloatTensor):
            pack = torch.tensor(pack)
        pack = pack.to(torch.float32)
        packs.append(pack)
        shapes.add(tuple(pack.shape))
    if len(shapes) != 1:
        raise RuntimeError(f"Voice packs have differing shapes: {shapes}")
    stacked = torch.stack(packs, dim=0)  # [N, ...]
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
    mixed = (w * stacked).sum(dim=0)
    return mixed.to(device)


def synth_line(pipe: KPipeline, mixed_pack: torch.Tensor, text: str, speed: float) -> np.ndarray:
    """Run pipeline, collect streaming chunks, return mono float32 at SR."""
    waves = []
    for res in pipe(text, voice=mixed_pack, speed=speed, split_pattern=SPLIT_PATTERN):
        if getattr(res, "audio", None) is None:
            continue
        waves.append(res.audio.detach().float().cpu().numpy())
    if not waves:
        raise RuntimeError("No audio returned from pipeline for text: " + repr(text[:80]))
    audio = np.concatenate(waves, axis=-1).astype("float32")
    return audio


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load text
    lines = read_lines(args.text_file)
    if not lines:
        raise SystemExit("No non-empty lines found in text_file.")

    # Parse voices & weights
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    weights = parse_weights(args.weights, voices)

    # Init pipeline & device
    pipe = KPipeline(lang_code=args.lang)
    device = pipe.model.device  # follow your GUI: keep everything on model's device

    # Prepare mixed pack once
    mixed_pack = load_and_blend_voices(pipe, voices, weights, device)

    # Decide zero padding for filenames
    pad = max(3, len(str(args.start_idx + len(lines) - 1)))

    # Write metadata for reproducibility
    meta_path = os.path.join(args.out_dir, args.metadata_csv)
    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf, delimiter="|")
        writer.writerow(["index", "filepath", "transcript", "text", "duration_sec", "has_speech", "voices", "weights_normalized", "speed", "lang", "sr"])
        for i, text in enumerate(tqdm(lines, desc="Synthesizing", unit="utt"), start=args.start_idx):
            if not text.strip():
                continue
            slug = safe_slug(text)
            fname = f"{args.prefix}_{str(i).zfill(pad)}_{slug}.wav"
            out_wav = os.path.join(args.out_dir, fname)
            try:
                audio = synth_line(pipe, mixed_pack, text, speed=args.speed)
                sf.write(out_wav, audio, SR)
                duration_sec = float(len(audio) / SR)
                relpath = os.path.relpath(out_wav, args.out_dir)
            except Exception as e:
                # Log the error and keep going
                err_name = f"{args.prefix}_{str(i).zfill(pad)}_ERROR.txt"
                with open(os.path.join(args.out_dir, err_name), "w", encoding="utf-8") as ef:
                    ef.write(f"Text:\n{text}\n\nError:\n{repr(e)}\n")
                tqdm.write(f"[!] Failed line {i}: {e}")
                continue

            writer.writerow([
                i,
                relpath,                   # filepath relative to audio root (out_dir)
                text,                      # transcript
                text,                      # text (duplicate for convenience)
                f"{duration_sec:.6f}",     # duration_sec
                "true",                    # has_speech
                ",".join(voices),
                ",".join([f"{w:.6f}" for w in weights]),
                f"{args.speed:.3f}",
                args.lang,
                SR,
            ])
            tqdm.write(f"[OK] {out_wav}")

    tqdm.write(f"\nDone. Wrote WAVs to: {args.out_dir}\nMetadata: {meta_path}")


if __name__ == "__main__":
    main()
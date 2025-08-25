#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare TTS training data from a folder of MP3s (macOS/MPS-friendly).

Pipeline:
1) VAD (WebRTC) to quickly detect speech presence.
2) Whisper transcription (runs on MPS if available).
3) Save:
   - Per-file transcript .txt
   - Per-file segments .csv
   - Corpus-level metadata.csv (common for TTS training)

Usage:
python prep_tts_from_mp3.py --audio_dir /path/to/mp3s --out_dir /path/to/out --model small
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import pandas as pd
from tqdm import tqdm

import torch
import whisper


# ----------------------------- Audio / VAD helpers -----------------------------

def load_audio_16k_pcm(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16kHz; return (samples, sr)."""
    # librosa handles mp3; resample to 16k mono
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    return y.astype(np.float32), 16000


def float_to_int16_pcm(y: np.ndarray) -> bytes:
    """Convert float32 (-1..1) to int16 PCM bytes."""
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)
    return y_int16.tobytes()


def frame_generator(audio_bytes: bytes, sample_rate: int, frame_ms: int = 30):
    """Yield 10/20/30ms frames for webrtcvad (int16 mono PCM)."""
    n_bytes_per_sample = 2  # int16
    frame_len = int(sample_rate * (frame_ms / 1000.0)) * n_bytes_per_sample
    for start in range(0, len(audio_bytes), frame_len):
        end = start + frame_len
        if end > len(audio_bytes):
            break
        yield audio_bytes[start:end]


def vad_speech_ratio(y: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    """
    Return fraction of frames flagged as speech by WebRTC VAD.
    aggressiveness: 0 (least) .. 3 (most aggressive filtering)
    """
    vad = webrtcvad.Vad(aggressiveness)
    pcm = float_to_int16_pcm(y)
    frames = list(frame_generator(pcm, sr, frame_ms=30))  # 30ms frames are robust
    if not frames:
        return 0.0
    speech_flags = [vad.is_speech(f, sr) for f in frames]
    return sum(speech_flags) / len(speech_flags)


def get_duration_sec(path: Path) -> float:
    """Fast duration via soundfile; fallback to librosa if needed."""
    try:
        with sf.SoundFile(str(path)) as f:
            return len(f) / float(f.samplerate)
    except Exception:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return len(y) / float(sr)


# ----------------------------- Whisper helpers --------------------------------

def pick_device() -> str:
    """Prefer MPS on Apple Silicon, else CUDA if available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transcribe_whisper(
    model,
    audio_path: Path,
    device: str,
    language: Optional[str] = None,
) -> dict:
    """
    Run Whisper transcription.
    Returns whisper result dict with 'text', 'segments', and 'language'.
    """
    # fp16 on cuda only (disable on mps to avoid unsupported ops)
    fp16 = (device == "cuda")
    result = model.transcribe(
        str(audio_path),
        language=language,        # None = auto
        verbose=False,
        fp16=fp16,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    return result


# ----------------------------- Text helpers -----------------------------------

def normalize_text(s: str) -> str:
    """Light normalization suitable for many TTS pipelines."""
    s = s.strip()
    s = " ".join(s.split())  # collapse whitespace
    # Optionally: lowercase for training consistency (comment out if you want case kept)
    # s = s.lower()
    return s


# ----------------------------- Main pipeline ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="data/JarvisSounds", type=str, help="Folder with MP3s (searched recursively).")
    ap.add_argument("--out_dir", default="data/process_JarvisSounds", type=str, help="Output folder for transcripts and metadata.")
    ap.add_argument("--model", default="base", type=str, help="Whisper size: tiny|base|small|medium|large")
    ap.add_argument("--speaker_source", choices=["fixed", "parent"], default="fixed",
                    help="Use 'fixed' single speaker label or take parent folder name as speaker.")
    ap.add_argument("--speaker_label", default="Jarvis", type=str, help="Speaker name if speaker_source=fixed.")
    ap.add_argument("--min_speech_ratio", type=float, default=0.05,
                    help="Minimum VAD speech fraction to consider file as 'has_speech'.")
    ap.add_argument("--extensions", nargs="+", default=[".mp3", ".wav", ".m4a", ".flac"], help="Audio extensions.")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    audio_files: List[Path] = []
    for ext in args.extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        print(f"No audio files with {args.extensions} under {audio_dir}", file=sys.stderr)
        sys.exit(1)

    # Load Whisper model on best device (fallback to CPU if MPS has unsupported ops)
    device = pick_device()
    print(f"[INFO] Using device: {device}")
    try:
        model = whisper.load_model(args.model, device=device)
    except NotImplementedError as e:
        print(f"[WARN] Whisper couldn't initialize on {device} ({e.__class__.__name__}: {e}). Falling back to CPU.")
        device = "cpu"
        model = whisper.load_model(args.model, device=device)

    meta_rows = []

    for path in tqdm(audio_files, desc="Processing audio"):
        try:
            duration = get_duration_sec(path)

            # Speech presence via VAD (quick)
            y16, sr16 = load_audio_16k_pcm(path)
            speech_ratio = vad_speech_ratio(y16, sr16, aggressiveness=2)
            has_speech = speech_ratio >= args.min_speech_ratio

            # Speaker label
            if args.speaker_source == "parent":
                speaker = path.parent.name or "spk"
            else:
                speaker = args.speaker_label

            rel_path = os.path.relpath(path, audio_dir)
            base = path.stem

            # File-specific outputs
            per_file_txt = out_dir / f"{base}.txt"
            per_file_segments = out_dir / f"{base}.segments.csv"
            nospeech_marker = out_dir / f"{base}.nospeech.txt"

            transcript = ""
            language = ""

            if not has_speech:
                # Save marker + empty transcript file for clarity
                nospeech_marker.write_text("NO_SPEECH_DETECTED\n")
                per_file_txt.write_text("")
            else:
                # Transcribe
                result = transcribe_whisper(model, path, device=device)
                language = result.get("language") or ""
                transcript = normalize_text(result.get("text", ""))

                # Save transcript .txt
                per_file_txt.write_text(transcript)

                # Save segments CSV
                segs = result.get("segments", []) or []
                if segs:
                    seg_rows = []
                    for s in segs:
                        seg_rows.append({
                            "start": float(s.get("start", 0.0)),
                            "end": float(s.get("end", 0.0)),
                            "text": normalize_text(s.get("text", "")),
                        })
                    pd.DataFrame(seg_rows).to_csv(per_file_segments, index=False)

            # Add to corpus metadata
            meta_rows.append({
                "filepath": str(rel_path),
                "basename": base,
                "duration_sec": round(float(duration), 3),
                "language": language,
                "speaker": speaker,
                "has_speech": bool(has_speech),
                "transcript": transcript,
            })

        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}", file=sys.stderr)

    # Save metadata.csv (TTS-friendly)
    meta_df = pd.DataFrame(meta_rows)
    # Common TTS format is also `metadata.txt` with "path|transcript" lines.
    # We provide a rich CSV, which you can easily convert later.
    meta_csv = out_dir / "metadata.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"[DONE] Wrote {meta_csv} with {len(meta_df)} rows.")
    print("Per-file transcripts and segment CSVs are in the same out_dir.")


if __name__ == "__main__":
    main()
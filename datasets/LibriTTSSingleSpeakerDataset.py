# datasets/LibriTTSSingleSpeakerDataset.py
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import librosa

from kokoro import KPipeline

SR = 24000
DIGITS_ONLY = re.compile(r"^\d+$")

def _extract_speaker_id(root: Path, wav: Path) -> str:
    rel = wav.relative_to(root)
    # first numeric directory under root is speaker id (LibriTTS-style)
    for part in rel.parts[:-1]:
        if DIGITS_ONLY.match(part):
            return part
    # fallback to first component
    return rel.parts[0] if len(rel.parts) > 0 else "UNKNOWN"

class LibriTTSSingleSpeakerDataset(Dataset):
    """
    Loads a *single speaker* from a LibriTTS-R / LibriTTS-P style tree:
        root/<speaker_id>/<chapter_or_subdir>/*.wav
        with paired "*.normalized.txt" next to each wav.

    Returns dict:
      - wav: FloatTensor [T]
      - text: str (normalized transcript)
      - ps: str (phoneme string via KPipeline G2P)
      - voice_pack: FloatTensor (KPipeline voice embedding, constant across items)
      - sr: int
      - path: Path (optional meta)
      - speaker_id: str
    """
    def __init__(
        self,
        root_dir: Path,
        speaker_id: str,
        lang_code: str = "a",
        sample_rate: int = SR,
        max_phonemes: int = 510,
        min_dur: float = 0.7,
        max_dur: float = 22.0,
        cache_g2p: bool = True,
        voice: str = "af_heart",
        quiet_pipeline: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir).resolve()
        self.speaker_id = str(speaker_id)
        self.sample_rate = sample_rate
        self.max_phonemes = max_phonemes
        self.cache_g2p = cache_g2p

        # Index wavs for only this speaker
        items: List[Tuple[Path, Path, str]] = []
        for wav in self.root_dir.rglob("*.wav"):
            spk = _extract_speaker_id(self.root_dir, wav)
            if spk != self.speaker_id:
                continue
            txt = wav.with_suffix(".normalized.txt")
            if not txt.exists():
                continue
            # Optional duration check via soundfile.info (fast) — or load later
            try:
                info = sf.info(str(wav))
                dur = float(info.frames) / float(info.samplerate)
            except Exception:
                dur = None
            if dur is not None and (dur < min_dur or dur > max_dur):
                continue
            items.append((wav, txt, spk))

        if shuffle:
            random.shuffle(items)

        if not items:
            raise RuntimeError(f"No items found for speaker_id={self.speaker_id} under {self.root_dir}")

        self.items = items

        # KPipeline for G2P + voice pack
        self.pipe = KPipeline(lang_code=lang_code, model=not quiet_pipeline)
        self.voice_pack = self.pipe.load_voice(voice)

        self._g2p_cache: Dict[int, str] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _text_to_phonemes(self, text: str, idx: int) -> str:
        if self.cache_g2p and idx in self._g2p_cache:
            return self._g2p_cache[idx]
        _, tokens = self.pipe.g2p(text)
        ps = ''.join(
            t.phonemes + (' ' if t.whitespace else '')
            for t in tokens if t.phonemes is not None
        ).strip()
        if len(ps) > self.max_phonemes:
            warnings.warn(f"Truncating phonemes from {len(ps)} to {self.max_phonemes}")
            ps = ps[:self.max_phonemes]
        if self.cache_g2p:
            self._g2p_cache[idx] = ps
        return ps

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav_path, txt_path, spk = self.items[idx]
        # Load audio
        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        wav = wav.astype("float32")

        # Read normalized text
        try:
            text = Path(txt_path).read_text(encoding="utf-8").strip()
        except Exception:
            text = ""

        ps = self._text_to_phonemes(text, idx)
        return {
            "wav": torch.from_numpy(wav),
            "text": text,
            "ps": ps,
            "voice_pack": self.voice_pack,
            "sr": self.sample_rate,
            "path": wav_path,
            "speaker_id": spk,
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(x["wav"].shape[0] for x in batch)
    wavs = torch.stack([F.pad(x["wav"], (0, max_len - x["wav"].shape[0])) for x in batch], dim=0)  # [B, T]
    texts = [x["text"] for x in batch]
    pss = [x["ps"] for x in batch]
    pack = batch[0]["voice_pack"]
    sr = batch[0]["sr"]
    return {
        "wav": wavs,
        "texts": texts,
        "pss": pss,
        "voice_pack": pack,
        "sr": sr,
        # Optional convenience for debugging:
        # "paths": [x["path"] for x in batch],
        # "speaker_ids": [x["speaker_id"] for x in batch],
    }
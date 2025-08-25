import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from kokoro import KPipeline
from typing import List, Dict, Any, Tuple, Optional
import  warnings
import soundfile as sf
import librosa


SR = 24000  # consistent sample rate for training + logging

class KokoroJarvisSFTDataset(Dataset):
    """
    Loads (audio, transcript) pairs from metadata.csv and produces phoneme strings via KPipeline G2P.
    """
    def __init__(self,
                 audio_root: Path,
                 metadata_csv: Path,
                 lang_code: str = "a",     # 'a' = American English
                 sample_rate: int = SR,
                 max_phonemes: int = 510,
                 min_dur: float = 0.7,
                 max_dur: float = 22.0,
                 cache_g2p: bool = True,
                 voice: str = "af_heart",
                 quiet_pipeline: bool = True):
        super().__init__()
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.max_phonemes = max_phonemes
        self.voice = voice

        df = pd.read_csv(metadata_csv, keep_default_na=False)
        rows = []
        for _, r in df.iterrows():
            if not str(r.get("has_speech", True)).lower() in ("true", "1"):
                continue
            val = r.get("transcript", "")
            if pd.isna(val): val = ""
            text = str(val).strip() or str(r.get("text", "")).strip()
            if not text:
                continue
            dur = float(r.get("duration_sec", 0))
            if dur < min_dur or dur > max_dur:
                continue
            wav_path = self.audio_root / r["filepath"]
            if wav_path.suffix.lower() not in (".wav", ".mp3", ".flac", ".m4a"): continue
            if not wav_path.exists(): continue
            rows.append((wav_path, text))
        self.items: List[Tuple[Path, str]] = rows

        # Initialize KPipeline with no model (quiet) — only for G2P + voice pack
        self.pipe = KPipeline(lang_code=lang_code, model=not quiet_pipeline)
        # Pre-load voice pack once
        self.voice_pack = self.pipe.load_voice(voice)

        self.cache_g2p = cache_g2p
        self._g2p_cache: Dict[int, str] = {}

        if len(self.items) == 0:
            raise RuntimeError("No training rows found. Check metadata_csv filters and paths.")

    def __len__(self):
        return len(self.items)

    def _text_to_phonemes(self, text: str, idx: int) -> str:
        if self.cache_g2p and idx in self._g2p_cache:
            return self._g2p_cache[idx]
        _, tokens = self.pipe.g2p(text)
        ps = ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens if t.phonemes is not None).strip()
        if len(ps) > self.max_phonemes:
            warnings.warn(f"Truncating phonemes from {len(ps)} to {self.max_phonemes}")
            ps = ps[:self.max_phonemes]
        if self.cache_g2p:
            self._g2p_cache[idx] = ps
        return ps

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, text = self.items[idx]
        wav, sr = sf.read(str(path))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        wav = wav.astype("float32")
        ps = self._text_to_phonemes(text, idx)
        return {"wav": torch.from_numpy(wav), "text": text, "ps": ps,
                "voice_pack": self.voice_pack, "sr": self.sample_rate}

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(x["wav"].shape[0] for x in batch)
    wavs = torch.stack([F.pad(x["wav"], (0, max_len - x["wav"].shape[0])) for x in batch], dim=0)  # [B, T]
    texts = [x["text"] for x in batch]
    pss = [x["ps"] for x in batch]
    pack = batch[0]["voice_pack"]
    sr = batch[0]["sr"]
    return {"wav": wavs, "texts": texts, "pss": pss, "voice_pack": pack, "sr": sr}
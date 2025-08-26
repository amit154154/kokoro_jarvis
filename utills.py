import torch
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path
from kokoro.model import KModel
from kokoro import KPipeline
import torch.nn.functional as F

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def str2bool_default_true(s):
    return False if isinstance(s, str) and s.lower() in ("0", "false", "no", "off") else True

def str2bool_default_false(s):
    return False if isinstance(s, str) and s.lower() in ("0", "false", "no", "off") else True



def list_voice_pt_files(voices_dir: Path) -> List[Path]:
    return sorted(voices_dir.glob("*.pt"))


def load_base_packs(voices_dir: Path) -> Tuple[List[torch.FloatTensor], List[str]]:
    pts = list_voice_pt_files(voices_dir)
    if not pts:
        raise RuntimeError(f"No .pt voice packs found under {voices_dir}")
    print(f"[init] using {len(pts)} local packs from {voices_dir}:")
    print("       " + ", ".join(p.stem for p in pts))
    packs, names = [], []
    shape = None
    for p in pts:
        t = torch.load(str(p), weights_only=True).float()
        if shape is None:
            shape = tuple(t.shape)
        elif tuple(t.shape) != shape:
            raise RuntimeError(f"Shape mismatch: {p} has {tuple(t.shape)} vs {shape}")
        packs.append(t)
        names.append(p.stem)
    return packs, names

def average_pack(packs: List[torch.FloatTensor]) -> torch.FloatTensor:
    stacked = torch.stack(packs, dim=0)  # [N, T, 1, D]
    return stacked.mean(dim=0)           # [T, 1, D]


def ref_slice_by_len(pack: torch.FloatTensor, ps: str) -> torch.FloatTensor:
    idx = max(1, min(len(ps), pack.size(0))) - 1
    ref = pack[idx]  # [1,D]
    return ref.unsqueeze(0) if ref.dim() == 1 else ref









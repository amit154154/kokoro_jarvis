#!/usr/bin/env python3
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

# All known American English voices (prefix af_ / am_)
US_VOICES = [
    "af_alloy", "af_bella", "af_betty", "af_candy", "af_daisy",
    "af_emma", "af_glinda", "af_heart", "af_jessica", "af_luna",
    "af_maya", "af_nicole", "af_rachel", "af_sarah", "af_tina",
    "af_vivian",
    "am_adam", "am_bob", "am_charlie", "am_daniel", "am_ethan",
    "am_frank", "am_george", "am_henry", "am_isaac", "am_jack",
    "am_kevin", "am_leo", "am_michael", "am_nick", "am_oscar",
    "am_paul", "am_quinn", "am_ryan", "am_sam", "am_tom",
    "am_victor", "am_will"
]

REPO_ID = "hexgrad/Kokoro-82M"
OUT_DIR = Path("kokoro_us_voices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name in US_VOICES:
    print(f"Downloading {name}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"voices/{name}.pt"
    )
    shutil.copy(local_path, OUT_DIR / f"{name}.pt")

print(f"✅ Downloaded {len(US_VOICES)} US voices into {OUT_DIR.resolve()}")
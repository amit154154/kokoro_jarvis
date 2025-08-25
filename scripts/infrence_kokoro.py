# kokoro_mps_demo.py
from kokoro import KPipeline
import soundfile as sf

def synth(text: str, out_wav: str, voice: str = "af_heart", speed: float = 1.0):
    """
    Generate speech with Kokoro and save a 24kHz WAV.
    Voices to try: af_heart, af_alloy, am_adam, am_metal, bf_anna, bm_lewis, etc.
    """
    # 'a' = American English; 'b' = British English
    pipe = KPipeline(lang_code="a")

    # Kokoro can split long text into chunks. We take the first (or stitch all).
    generator = pipe(text, voice=voice, speed=speed, split_pattern=r"\n+")
    audio_24k = bytearray()
    sr = 24000

    # Collect all chunks (you can also save each chunk separately)
    for _, _, audio in generator:
        # audio is a float32 numpy array at 24kHz
        sf.write(out_wav, audio, sr)  # overwrite with last chunk, or:
        # If you want to concatenate, collect then write once:
        # audio_24k.extend(audio.tobytes())

    # Example to stitch:
    # import numpy as np
    # if audio_24k:
    #     merged = np.frombuffer(bytes(audio_24k), dtype="float32")
    #     sf.write(out_wav, merged, sr)

if __name__ == "__main__":
    synth(
        text="Hello! This is Kokoro running on Apple Silicon with MPS.",
        out_wav="kokoro_demo.wav",
        voice="af_heart",
        speed=1.0,
    )
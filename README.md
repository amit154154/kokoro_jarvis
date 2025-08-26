# Kokoro Fine-Tuning and Experimentation

This project is an ongoing **research and experimentation framework** around [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — a super small, efficient text-to-speech (TTS) model.  
The focus is on **fine-tuning**, **optimizing and exploring voice embeddings**, and investigating future directions for adapting Kokoro to new speakers, styles, and research applications.

## Optimizing Kokoro Voices

The goal is to optimize voice embeddings for a dataset consisting of WAV files and their corresponding transcripts.  
I explore two optimization strategies:

### Small Adjustments

This method optimizes a **low-dimensional vector of coefficients** over existing Kokoro voice embeddings.  
The coefficients act as mixing weights, combining multiple pretrained voices into a new embedding.  
This provides a lightweight way to adapt Kokoro to a target voice without fully retraining the model.

#### Results
<details>
<summary><b>Show coefficient trajectory & audio samples</b></summary>

<br>

<div align="center">

<div align="center">
  <img src="assets/optemize_voices/coeff_viz.gif" alt="Coefficient trajectory (animated)" width="100%"/><br/>
  <sub>
    Training path of the mixture weights in the circular embedding space.
    &nbsp;•&nbsp;
    <a href="assets/optemize_voices/coeff_viz.mp4">Download MP4</a>
  </sub>
</div>

**Audio Samples**  
_GitHub tip: the README preview doesn’t always render audio players. Click a link below, then press **“View raw”** to play in your browser._

- ▶️ **Start Generation** — [WAV](assets/optemize_voices/start_generation.wav)
- ▶️ **Optimized Generation** — [WAV](assets/optemize_voices/optemize_generation.wav)
- ▶️ **Ground Truth** — [WAV](assets/optemize_voices/ground_truth.wav)

</div>

<p align="center">
  <img src="assets/optemize_voices/loss_figure.png" width="500"/>
  <br><em>Loss curve showing convergence during coefficient optimization.</em>
</p>

</details>

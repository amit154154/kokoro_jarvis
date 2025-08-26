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

<!-- Preview on the left, audio on the right -->
<div align="center">

<table>
<tr>
  <td align="center" width="55%">

    <!-- Animated preview looks great on GitHub -->
    <img src="assets/optemize_voices/coeff_viz.gif" alt="Coefficient trajectory (animated)" width="100%"/>

    <sub>
    Training path of the mixture weights in the circular embedding space.<br>
    <a href="assets/optemize_voices/coeff_viz.mp4">Download MP4</a>
    </sub>

  </td>
  <td align="left" width="45%">

    <b>Audio Samples</b><br><br>

    <div>
      <b>Start Generation</b><br>
      <audio controls src="assets/optemize_voices/start_generation.wav"></audio>
    </div>
    <br>

    <div>
      <b>Optimized Generation</b><br>
      <audio controls src="assets/optemize_voices/optemize_generation.wav"></audio>
    </div>
    <br>

    <div>
      <b>Ground Truth</b><br>
      <audio controls src="assets/optemize_voices/ground_truth.wav"></audio>
    </div>

  </td>
</tr>
</table>

</div>

<p align="center">
  <img src="assets/optemize_voices/loss_figure.png" width="680" alt="Training loss curve"/><br>
  <em>Loss curve showing convergence during coefficient optimization.</em>
</p>

</details>
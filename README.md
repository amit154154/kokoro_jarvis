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

<table>
<tr>
  <td align="center"><b>Coefficient Trajectory</b></td>
  <td align="center"><b>Audio Samples</b></td>
</tr>
<tr>
  <td>
    <video src="assets/optemize_voices/coeff_viz.mp4" controls width="400"></video>
    <br>
    <em>Training path of the mixture weights in the circular embedding space.</em>
  </td>
  <td>
    <p><b>Start Generation</b></p>
    <audio controls src="assets/optemize_voices/start_generation.wav"></audio>
    <br><br>
    <p><b>Optimized Generation</b></p>
    <audio controls src="assets/optemize_voices/optemize_generation.wav"></audio>
    <br><br>
    <p><b>Ground Truth</b></p>
    <audio controls src="assets/optemize_voices/ground_truth.wav"></audio>
  </td>
</tr>
</table>

<p align="center">
  <img src="assets/optemize_voices/loss_figure.png" alt="Training Loss" width="500"/>
  <br>
  <em>Loss curve showing convergence during coefficient optimization.</em>
</p>
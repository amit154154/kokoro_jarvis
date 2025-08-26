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

| Visualization                                                                                                                                      | Audio Samples                                                                                                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🎥 [Coefficient Trajectory](assets/optemize_voices/coeff_viz.mp4) <br> _The training path of the mixture weights in the circular embedding space._ | 🔊 [Start Generation](assets/optemize_voices/start_generation.wav) <br> _Before optimization_ <br><br> 🔊 [Optimized Generation](optemize_voices/optemize_generation.wav) <br> _After optimization_ <br><br> 🎙️ [Ground Truth](assets/optemize_voices/ground_truth.wav) <br> _Target voice_ |

<p align="center">
  <img src="assets/optemize_voices/loss_figure.png" alt="Training Loss" width="500"/>
  <br>
  <em>Loss curve showing convergence during coefficient optimization.</em>
</p>
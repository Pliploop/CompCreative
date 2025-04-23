---
layout: default
title: ECS7022P Project Companion
---

# Computational Creativity – Project Companion

Welcome to the companion site for my ECS7022P Computational Creativity assignment. This project investigates how generative AI can support **co-creative music workflows**, especially in sample-based music and DJing contexts.

Rather than generating full tracks, this system aims to empower artists with controllable, iterative generation of **audio samples** via **text prompts**.

---

## 🎯 Project Overview

This project explores multimodal generation using diffusion-based audio models. It focuses on:

- Text-to-audio generation with corresponding spectrograms
- User prompt control and creative input
- DDIM sampling experiments to analyze variation and fidelity

The system emphasizes **co-creation** rather than full automation.

---

## 🧠 Multimodal Generation

Below are examples of audio generated from text prompts. Each includes:

- 📝 A prompt
- 🎵 A generated audio sample
- 📊 Its corresponding spectrogram

### 🔹 Example 1

**Prompt:** _"A dreamy synth loop with a mellow vibe, suitable for lo-fi study sessions."_

<img src="/examples/spectrogram1.png" alt="Spectrogram 1" width="500" />
<audio controls>
  <source src="/examples/audio1.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

---

### 🔹 Example 2

**Prompt:** _"Percussive glitch textures paired with airy ambient pads."_

<img src="/examples/spectrogram2.png" alt="Spectrogram 2" width="500" />
<audio controls>
  <source src="/examples/audio2.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

---

## ⏱ DDIM Experiments

The Denoising Diffusion Implicit Models (DDIM) method allows more controlled sampling. This section explores:

- Sampling with fewer steps for speed
- Sampling with more steps for fidelity
- Visual and auditory comparisons across variations

### 🔹 Example A – Fewer Steps

**Prompt:** _"Vinyl-crackled jazz loop with electric piano and brushed drums."_

<img src="/ddim/spectrogram_fast.png" alt="DDIM Fast Sampling" width="500" />
<audio controls>
  <source src="/ddim/audio_fast.wav" type="audio/wav">
</audio>

---

### 🔹 Example B – More Steps

**Same prompt, but sampled with 200 DDIM steps.**

<img src="/ddim/spectrogram_slow.png" alt="DDIM Full Sampling" width="500" />
<audio controls>
  <source src="/ddim/audio_slow.wav" type="audio/wav">
</audio>

---

## 📎 Resources

- [GitHub Repository](https://github.com/yourusername/your-repo-name)
- [Colab Notebook](https://colab.research.google.com/your-notebook-link)
- [PDF Report (optional)](report.pdf)

---

_This site was generated using Jekyll and hosted with GitHub Pages._

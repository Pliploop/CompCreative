---
layout: default
title: ECS7022P Project Companion - Julien Guinot
---

# Computational Creativity – Project Companion

Welcome to the companion site for my ECS7022P Computational Creativity assignment. This project investigates how generative AI can support **co-creative music workflows**, especially in sample-based music and DJing contexts.

Rather than generating full tracks, this system empowers artists with controllable, iterative generation of **audio samples** via **text prompts**, **inversion**, and **direct latent manipulation**.

---

## 🎨 Conditional Text-to-Audio Generation

Explore controllable music generation from text prompts.

### 🔹 Example 1
**Prompt:** _"ambient, piano, love"_

<img src="/samples/cond1.png" alt="Spectrogram 1" width="500" />
<audio controls>
  <source src="/samples/cond1.wav" type="audio/wav">
</audio>

### 🔹 Example 2
**Prompt:** _"glitchy, industrial, techno"_

<img src="/samples/cond2.png" alt="Spectrogram 2" width="500" />
<audio controls>
  <source src="/samples/cond2.wav" type="audio/wav">
</audio>

### 🔹 Example 3
**Prompt:** _"funk, bass, upbeat"_

<img src="/samples/cond3.png" alt="Spectrogram 3" width="500" />
<audio controls>
  <source src="/samples/cond3.wav" type="audio/wav">
</audio>

---

## 🔁 DDIM Inversion – Prompt-Based Editing

Transform a generated audio sample by modifying its prompt. Each row below shows:

- Original prompt → New prompt
- Spectrogram of edited output
- Audio of edited output

### 🔹 Example 1
**Original Prompt:** _"lofi, guitar, mellow"_ → **Edited Prompt:** _"lofi, guitar, distorted"_

<img src="/samples/invert1.png" alt="Spectrogram Invert 1" width="500" />
<audio controls>
  <source src="/samples/invert1.wav" type="audio/wav">
</audio>

### 🔹 Example 2
**Original Prompt:** _"jazz, saxophone"_ → **Edited Prompt:** _"jazz, saxophone, ambient"_

<img src="/samples/invert2.png" alt="Spectrogram Invert 2" width="500" />
<audio controls>
  <source src="/samples/invert2.wav" type="audio/wav">
</audio>

### 🔹 Example 3
**Original Prompt:** _"drum and bass, club, energetic"_ → **Edited Prompt:** _"breakcore, distorted, chaos"_

<img src="/samples/invert3.png" alt="Spectrogram Invert 3" width="500" />
<audio controls>
  <source src="/samples/invert3.wav" type="audio/wav">
</audio>

---

## 🧠 Inversion Variants – Audio-Guided and Negative Prompt Edits

### 🎙 Audio-to-Audio Guidance
Use one audio sample to guide another using DDIM inversion.

### 🔹 Example 1
_Source → Target audio guidance (semantic morphing)_

<img src="/samples/audio_guided1.png" alt="Spectrogram Audio Guided 1" width="500" />
<audio controls>
  <source src="/samples/audio_guided1.wav" type="audio/wav">
</audio>

### 🔹 Example 2

<img src="/samples/audio_guided2.png" alt="Spectrogram Audio Guided 2" width="500" />
<audio controls>
  <source src="/samples/audio_guided2.wav" type="audio/wav">
</audio>

### 🔹 Example 3

<img src="/samples/audio_guided3.png" alt="Spectrogram Audio Guided 3" width="500" />
<audio controls>
  <source src="/samples/audio_guided3.wav" type="audio/wav">
</audio>

### 🚫 Negative Prompt Inversion
Guide the model **away** from a concept (e.g. "remove vocals").

### 🔹 Example 1
**Prompt:** _"jazz, vocals"_ → **Negative Prompt:** _"vocals"_

<img src="/samples/neg1.png" alt="Negative Inversion 1" width="500" />
<audio controls>
  <source src="/samples/neg1.wav" type="audio/wav">
</audio>

### 🔹 Example 2
**Prompt:** _"orchestral, cinematic"_ → **Negative Prompt:** _"cinematic"_

<img src="/samples/neg2.png" alt="Negative Inversion 2" width="500" />
<audio controls>
  <source src="/samples/neg2.wav" type="audio/wav">
</audio>

### 🔹 Example 3
**Prompt:** _"vocal chop, club"_ → **Negative Prompt:** _"vocal"_

<img src="/samples/neg3.png" alt="Negative Inversion 3" width="500" />
<audio controls>
  <source src="/samples/neg3.wav" type="audio/wav">
</audio>

---

## 🎛 Deterministic Latent Manipulations

Latent-level transformations that are fast, repeatable, and interpretable.

### ✨ Vector-Style Latent Operations
Examples: Interpolation, blending, mirroring

### 🔹 Example 1 – Interpolation between A and B
<img src="/samples/latent_vec1.png" alt="Latent Interpolation" width="500" />
<audio controls>
  <source src="/samples/latent_vec1.wav" type="audio/wav">
</audio>

### 🔹 Example 2 – Flip Channels
<img src="/samples/latent_vec2.png" alt="Flip Channels" width="500" />
<audio controls>
  <source src="/samples/latent_vec2.wav" type="audio/wav">
</audio>

### 🔹 Example 3 – Crossfade
<img src="/samples/latent_vec3.png" alt="Crossfade" width="500" />
<audio controls>
  <source src="/samples/latent_vec3.wav" type="audio/wav">
</audio>

---

### 🖼 Image-Style Latent Edits
Examples: Contrast, Jitter, Bloat, Warp

### 🔹 Example 1 – Adjusted Contrast + Warp
<img src="/samples/latent_img1.png" alt="Contrast Warp" width="500" />
<audio controls>
  <source src="/samples/latent_img1.wav" type="audio/wav">
</audio>

### 🔹 Example 2 – Bloat + Channel Permutation
<img src="/samples/latent_img2.png" alt="Bloat" width="500" />
<audio controls>
  <source src="/samples/latent_img2.wav" type="audio/wav">
</audio>

### 🔹 Example 3 – Cutout + Jitter
<img src="/samples/latent_img3.png" alt="Cutout Jitter" width="500" />
<audio controls>
  <source src="/samples/latent_img3.wav" type="audio/wav">
</audio>

---

## 📎 Resources

- [GitHub Repository](https://github.com/Pliploop/CompCreative)
- [Colab Notebook](https://colab.research.google.com/drive/1aqACUO1xs5GzviQyLbem_MmnLE9V8Xwy?usp=sharing)
- [PDF Report](report.pdf)

_This site was generated using Jekyll and hosted with GitHub Pages._
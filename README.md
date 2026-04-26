# 🧠 Brain-to-Text

> *Can a machine read your thoughts?*
> This project decodes neural brain signals directly into spoken words — no voice, no movement, just brain activity.

---

## What is this?

When a person thinks of speaking a sentence, their brain fires electrical signals. This project takes those raw neural signals and converts them into text using deep learning.

Built on the **Brain-to-Text 2025** competition dataset from participant **T15** — 45 recording sessions spanning 2023–2025.

---

## How it works
Brain signals (512 electrodes)
        ↓
  Day-specific adapter        ← handles signal drift across sessions
        ↓
  5-layer GRU model           ← learns phoneme patterns over time
        ↓
  CTC decoding                ← converts sequence to phonemes
        ↓
  CMUdict + 4-gram LM         ← phonemes → real English words
        ↓
       Text
---

## Results

| Metric | Score |
|--------|-------|
| Phoneme Error Rate (PER) | **10.82%** |
| Word Error Rate (WER) | **18.3%** |
| Sessions tested | 45 |
| Total phonemes checked | 44,933 |
| Best session PER | **0.16%** |

---

## Problems we faced & how we solved them

**1. Brain signal drift across sessions**
Neural signals change day to day — same person, completely different signal distribution each session. We first tried z-score normalization but it actually made performance worse. The fix was a small learnable adapter layer per session (just a linear transform + bias) that lets the model adjust for each day without touching the shared GRU weights. Each of the 45 sessions gets its own adapter, everything else is shared.

**2. Building one universal model, not 45 separate ones**
The naive approach is to train a separate model per session. Instead we trained a single universal GRU that works across all 45 sessions simultaneously. Each batch randomly picks 4 sessions, takes 16 trials from each (64 trials total), and the model sees all sessions mixed together in every step. The day adapter handles the per-session differences, and the GRU learns what is common across all of them.

**3. CTC alignment — mismatched lengths between predictions and targets**
The GRU outputs a logit for every single timestep — for a typical trial that's 200+ output frames. But the ground truth is only 20-30 phonemes. There is no label telling the model which output frame corresponds to which phoneme. CTC solves this by summing over all possible alignments between the long output sequence and the short target sequence, so the model learns timing on its own without any manual alignment.

**4. Noisy raw neural signals**
Raw neural signals are very noisy. We applied a 5-frame Gaussian smoothing filter to every trial before feeding it into the model. This cleaned up the signal without losing the important temporal structure that the GRU needs to learn from.

---

## Notebooks

| File | What it does |
|------|-------------|
| `data.ipynb` | Load and explore the HDF5 neural data |
| `model_training.ipynb` | Build, train and validate the GRU model |
| `decode.ipynb` | Convert phoneme predictions → English words + WER evaluation |

---

## With more resources

This was built and trained with limited compute. With more resources the next steps would be replacing the GRU with a Transformer and swapping the 4-gram language model for a fine-tuned LLM. The architecture is already modular so any piece can be swapped independently.

---

## Built by

**Sanchit Luthra** — [@sanchitluthra](https://github.com/sanchitluthra)

Competition: [Brain-to-Text 2025](https://github.com/fwillett/speechBCI)

---

*The pipeline is 5 steps. Getting those 5 steps to work took 45 sessions worth of pain.*

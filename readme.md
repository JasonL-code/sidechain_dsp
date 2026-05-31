# Sidechain DSP (V1)

## Overview
This project is a multirate DSP engine that uses dynamic EQ to resolve frequency masking between two competing audio signals. The current version focuses on low-end conflicts between kick drum and bass.

## Key Designs

* **Block Processing:** To simulate real-time audio processing in an offline environment, the pipeline processes audio in blocks (512 samples/block) rather than the whole file. However, processing continuous audio streams in isolated blocks would cause phase discontinuities (clicks and pops) since an IIR filter requires past states to calculate the current sample. The engine explicitly passes these history states (`x_hist` and `y_hist`) across consecutive blocks to ensure seamless output.
* **Multirate Processing:** Envelope tracking, RMS estimation, and biquad coefficient updates are evaluated at a lower control rate (k-rate, 86.13 Hz), as they do not require sample-accurate resolution. The Direct Form I difference equation executes at the full sample rate (a-rate, 44.1 kHz). This separation also reduces interpreter overhead in native Python.

---

## Signal Flow

```
Drum ──→ Bandpass ──┐
         (60-120Hz) │ (a-rate)
                    ↓
┌────── k-rate (Block Level) ──────┐
│ RMS ──→ Gain Red. ──→ Envelope   │
└───────────────────┬──────────────┘
                    ↓ Coefs (b_n, a_n)
Bass ───────────→ Peaking EQ ──→ Output (a-rate)
```

## Known Limitations & V2 Roadmap

* **Static Threshold:** The `threshold_db` parameter is fixed. In trigger tracks with high background noise or long reverb tails, the detector can be falsely triggered, causing the target track to be continuously attenuated. This was confirmed in testing against an AI-separated stem from a commercial recording (*Giving Into the Love* — Aurora), where the dense drum arrangement and long reverb tails produced near-continuous false triggers.
* **Fixed EQ Center Frequency:** The `target_fc` parameter is static at 90Hz. However, different kick drums have distinct center frequencies, meaning a fixed value cannot generalize across kick drums with varying spectral profiles.

V2 will address these issues by introducing transient onset detection to replace RMS-based triggering, and an adaptive EQ center frequency derived from spectral centroid estimation.

## How to Run

### Prerequisites
* **Python Runtime:** Python >= 3.9 (Tested on native CPython framework)
* **Core Libraries:** `numpy`, `scipy`, `soundfile`, `matplotlib`

### Installation & Execution

```bash
pip install -r requirements.txt
```

Place stems in `sample1/`:

```
sample1/
  Drum.wav
  Bass.wav
```

The included stems are original compositions. Any mono or stereo WAV files at any sample rate will work — the pipeline resamples to 44100Hz on load.

```bash
python reproduce_results.py
```

Outputs land in `output/`:

- `bass_processed.wav` — Processed bass stem
- `ab_test_original_mix.wav` / `ab_test_processed_mix.wav` — peak-matched for A/B comparison
- `bass_spectrogram.png` — 3-panel logarithmic STFT chart displaying the original signal, the processed signal, and the Attenuation Footprint to visually verify the dynamic compression.

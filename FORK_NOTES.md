# runnable-anywhere fork

Changes against NVIDIA/diffusion-audio-restoration to make inference run on
an ordinary machine (tested: Apple Silicon, 16 GB, MPS). Upstream ships a
cluster-shaped Lightning setup: `accelerator: gpu` + DDP + a SLURM plugin,
placeholder checkpoint paths, a hardcoded `/debug` output dir, and imports
(`ssr_eval`, `moviepy`) that are validation-only but break inference imports.

## Changes

- **Guarded validation-only imports.** `ssr_eval` depends on the Python-2 era
  `mysql-python` and cannot be installed; `moviepy` 2.x removed the imported
  module path. Both are only used in validation, so the imports degrade
  gracefully.
- **Memory: keep only the final prediction.** `ddpm_sample` retained every
  diffusion step's full spectrogram and returned all of them, though callers
  use only the last — ~0.5 GB per step for a 7-minute track.
- **`restore.py`** — one-command inference. Downloads checkpoints, selects
  device, prepares input, assembles config, drives the unmodified
  `ensembled_inference_api.py` path.
- **`input_prep.py`** — cutoff-knee detection and brick-walling. The model is
  trained exclusively on brick-wall cutoffs (`UpsampleMask` zeroes whole FFT
  bins); real codec rolloffs are smeared over ~1 kHz, and given a cutoff at
  the *foot* of the rolloff the model reads the taper as natural spectrum and
  extends nothing. `restore.py` therefore finds the knee and walls the input
  there. Details and tests in the module.

## Setup

    ./setup.sh
    .venv/bin/python restore.py input.wav output.wav

Options: `--cutoff-hz N` (default: detect), `--no-brickwall` (input already
hard-cut), `--steps N` (default 20), `--single-split` (half the memory,
audibly worse — flat spectral shelf instead of a natural rolloff),
`--predict-batch-size N` (memory knob; upstream's default of 16 exhausts
16 GB MPS).

Checkpoints: the released weights are the 1-split model and the 2-split
ensemble (the paper's headline numbers use an unreleased 4-partition
ensemble). `restore.py` defaults to 2-split.

## Tests

    .venv/bin/python -m pytest tests/

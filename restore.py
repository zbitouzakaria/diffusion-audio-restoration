# ---------------------------------------------------------------
# One-command inference entry point.
# Added by the runnable-anywhere fork; not part of upstream.
# ---------------------------------------------------------------
"""Restore a band-limited audio file with A2SB.

    python restore.py input.wav output.wav [--cutoff-hz N] [--steps N]

Handles everything the cluster-shaped Lightning CLI leaves to the caller:
checkpoint download, device selection, input preparation (mono sum, cutoff
knee detection, brick-walling — see input_prep.py for why), and config
assembly. Drives the same `ensembled_inference_api.py` path as upstream, so
output is bit-identical to a hand-driven run with the same settings.

The model is mono and 44.1 kHz; a stereo file is restored one channel at a
time inside a single inference run (one checkpoint load), both channels
sharing the cutoff detected on their mono sum. Output has the input's channel
count, padded to the input's length.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from input_prep import bandwidth_hz, brickwall_lowpass

SAMPLE_RATE = 44_100
HF_REPO = "nvidia/audio_to_audio_schrodinger_bridge"
SINGLE_SPLIT = ("ckpt/A2SB_onesplit_0.0_1.0_release.ckpt",)
TWO_SPLIT = (
    "ckpt/A2SB_twosplit_0.0_0.5_release.ckpt",
    "ckpt/A2SB_twosplit_0.5_1.0_release.ckpt",
)
REPO_ROOT = Path(__file__).resolve().parent


def fetch_checkpoints(single_split: bool) -> list[str]:
    from huggingface_hub import hf_hub_download

    names = SINGLE_SPLIT if single_split else TWO_SPLIT
    return [hf_hub_download(HF_REPO, name) for name in names]


def build_config(
    filelist: list[dict],
    checkpoints: list[str],
    device: str,
    cutoff_hz: float,
    tmp_dir: Path,
) -> dict:
    return {
        "trainer": {
            "accelerator": device,
            "strategy": "auto",
            "devices": 1,
            "num_nodes": 1,
            "plugins": None,
            "logger": False,
        },
        # Upstream's CLI defaults this to /debug, which is unwritable outside
        # their containers.
        "checkpoint_callback": {"dirpath": str(tmp_dir / "lightning")},
        "model": {
            "pretrained_checkpoints": checkpoints,
            "t_cutoffs": [0.5] if len(checkpoints) == 2 else [],
        },
        "data": {
            "num_workers": 0,
            "batch_size": 1,
            "mix_dataset_config": {},
            "predict_filelist": filelist,
            "transforms_aug": [
                {
                    "class_path": "corruption.corruptions.MultinomialInpaintMaskTransform",
                    "init_args": {
                        "p_upsample_mask": 1.0,
                        "p_extension_mask": 0.0,
                        "p_inpaint_mask": 0.0,
                        "fill_noise_level": 0.5,
                        "sampling_rate": SAMPLE_RATE,
                        "upsample_mask_kwargs": {
                            "min_cutoff_freq": int(cutoff_hz),
                            "max_cutoff_freq": int(cutoff_hz),
                        },
                        "inpainting_mask_kwargs": {
                            "min_inpainting_frac": 0.1013,
                            "max_inpainting_frac": 0.1013,
                            "is_random": False,
                        },
                    },
                }
            ],
        },
    }


def restore(
    input_path: Path,
    output_path: Path,
    cutoff_hz: float | None = None,
    brickwall: bool = True,
    steps: int = 20,
    single_split: bool = False,
    device: str = "mps",
    predict_batch_size: int = 2,
) -> None:
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"A2SB expects {SAMPLE_RATE} Hz, got {sr} Hz.")
    channels = np.ascontiguousarray(audio.T)
    if channels.shape[0] > 1:
        print(f"restore: {channels.shape[0]} channels, restored one at a time")

    # One knee for the whole file: channels of one encode share a cutoff, and
    # detecting on the mono sum keeps them identical.
    if cutoff_hz is None:
        cutoff_hz = bandwidth_hz(channels.mean(axis=0), sr)
        print(f"restore: detected bandwidth knee at {cutoff_hz:.0f} Hz")
    else:
        print(f"restore: using given cutoff {cutoff_hz:.0f} Hz")

    checkpoints = fetch_checkpoints(single_split)
    print(f"restore: {len(checkpoints)}-split, {steps} steps, {device}")

    total = channels.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filelist = []
        for c, channel in enumerate(channels):
            if brickwall:
                channel = brickwall_lowpass(channel, sr, cutoff_hz)
            wav = tmp / f"in_c{c}.wav"
            sf.write(str(wav), channel, sr, subtype="FLOAT")
            filelist.append({"filepath": str(wav), "output_subdir": f"c{c}"})
        if brickwall:
            print(f"restore: brick-walled input at {cutoff_hz:.0f} Hz")

        out_dir = tmp / "out"
        config = tmp / "override.yaml"
        config.write_text(
            yaml.safe_dump(build_config(filelist, checkpoints, device, cutoff_hz, tmp))
        )
        result = subprocess.run(
            [
                sys.executable, "ensembled_inference_api.py", "predict",
                "-c", "configs/ensemble_2split_sampling.yaml",
                "-c", "configs/inference_files_upsampling.yaml",
                "-c", str(config),
                f"--model.predict_n_steps={steps}",
                f"--model.predict_batch_size={predict_batch_size}",
                f"--model.output_audio_filename={out_dir / 'recon.wav'}",
                "--model.output_per_input=true",
            ],
            cwd=REPO_ROOT,
        )
        recons = [out_dir / f"c{c}" / "recon.wav" for c in range(channels.shape[0])]
        if result.returncode != 0 or not all(r.exists() for r in recons):
            raise RuntimeError("A2SB inference produced no output; see log above.")
        out = np.zeros((channels.shape[0], total), dtype=np.float32)
        for c, recon in enumerate(recons):
            restored, _ = sf.read(str(recon), dtype="float32", always_2d=True)
            n = min(total, restored.shape[0])
            out[c, :n] = restored[:n, 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), out.T, sr, subtype="FLOAT")
    print(f"restore: wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cutoff-hz", type=float, default=None,
                        help="brick-wall/mask frequency; default: detect the knee")
    parser.add_argument("--no-brickwall", action="store_true",
                        help="input already has a hard cutoff; mask only")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--single-split", action="store_true",
                        help="1-split checkpoint: half the memory, audibly worse")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--predict-batch-size", type=int, default=2,
                        help="spectrogram windows per forward pass; memory knob only")
    args = parser.parse_args()
    restore(
        args.input, args.output,
        cutoff_hz=args.cutoff_hz,
        brickwall=not args.no_brickwall,
        steps=args.steps,
        single_split=args.single_split,
        device=args.device,
        predict_batch_size=args.predict_batch_size,
    )


if __name__ == "__main__":
    main()

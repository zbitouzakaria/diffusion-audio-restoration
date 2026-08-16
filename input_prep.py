# ---------------------------------------------------------------
# Input preparation for A2SB inference on real-world material.
# Added by the runnable-anywhere fork; not part of upstream.
# ---------------------------------------------------------------
"""A2SB is trained exclusively on brick-wall cutoffs.

`UpsampleMask` zeroes whole FFT bins, so every training example has a
perfectly sharp band edge. Real codec rolloffs are smeared over roughly a
kilohertz. Handed such an input with the cutoff set at the end of the taper,
the model reads the taper as a natural spectrum, concludes nothing is
missing, and extends nothing — measured on a 5 kHz-limited file, a cutoff of
4750 Hz produced silence above it while 4000 Hz reconstructed the full band.

The remedy implemented here: find the *knee* where the rolloff begins, and
brick-wall the input just there, giving the model the sharp edge it was
trained on. The taper region is sacrificed; it was already half-destroyed.
"""

import numpy as np


def bandwidth_hz(
    audio: np.ndarray,
    sample_rate: int,
    min_hz: float = 3000.0,
    drop_db: float = 20.0,
    band_hz: float = 250.0,
    span_hz: float = 1000.0,
) -> float:
    """Where a band-limited file starts falling off, in Hz — the knee.

    Deliberately not a spectral-rolloff percentile: in music almost all the
    energy sits below ~2 kHz, so a 99% rolloff reports ~2 kHz whatever the
    real bandwidth is (this is what `compute_rolloff_freq` in the upstream
    inference scripts does). The drop is measured across `span_hz` rather
    than between adjacent bands because real rolloffs are smeared and no
    250 Hz pair ever shows the full step.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)

    edges = np.arange(0, sample_rate / 2, band_hz)
    levels = np.array(
        [
            10 * np.log10(max(spectrum[(freqs >= lo) & (freqs < lo + band_hz)].sum(), 1e-30))
            for lo in edges
        ]
    )

    span = max(1, int(round(span_hz / band_hz)))
    start = max(int(min_hz // band_hz), span)
    if start >= levels.size:
        return sample_rate / 2
    drops = levels[start:] - levels[start - span : levels.size - span]
    if drops.size == 0 or drops.min() > -drop_db:
        return sample_rate / 2
    return float(edges[start + int(np.argmin(drops)) - span])


def brickwall_lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Zero everything at or above `cutoff_hz`, with no transition band.

    Produces the training-matched input: a wall, not a rolloff.
    """
    squeeze = audio.ndim == 1
    if squeeze:
        audio = audio[None, :]
    spectrum = np.fft.rfft(audio, axis=1)
    freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate)
    spectrum[:, freqs >= cutoff_hz] = 0.0
    out = np.fft.irfft(spectrum, audio.shape[1], axis=1).astype(np.float32)
    return out[0] if squeeze else out

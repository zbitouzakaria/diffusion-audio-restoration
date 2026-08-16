# ---------------------------------------------------------------
# Tests for the runnable-anywhere fork's input preparation.
# Run with: .venv/bin/python -m pytest tests/
# ---------------------------------------------------------------
"""Getting the cutoff wrong is not a small error.

A2SB regenerates everything above the cutoff it is told. An underestimate
makes it overwrite content that was already there; an overestimate (or a
cutoff placed at the foot of a smeared rolloff instead of the knee) makes it
treat the input as full-band and extend nothing.
"""

import numpy as np
import pytest

from input_prep import bandwidth_hz, brickwall_lowpass

SR = 44_100


def band_limited(cutoff_hz: float, seconds: float = 2.0) -> np.ndarray:
    """Pink-ish noise brick-walled at `cutoff_hz`, like a codec leaves it."""
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum /= np.maximum(freqs, 20.0)
    spectrum[freqs >= cutoff_hz] = 0.0
    signal = np.fft.irfft(spectrum, n).astype(np.float32)
    signal /= np.max(np.abs(signal))
    return np.tile(signal, (2, 1))


def smeared(cutoff_hz: float, width_hz: float = 1000.0, seconds: float = 4.0) -> np.ndarray:
    """A rolloff that fades over `width_hz` rather than stopping dead.

    What real codecs leave behind, and the case a brick-wall fixture does not
    cover: the drop is spread, so no pair of narrow adjacent bands shows it.
    """
    rng = np.random.default_rng(1)
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spectrum /= np.maximum(freqs, 20.0)
    taper = np.clip((cutoff_hz + width_hz - freqs) / width_hz, 0.0, 1.0)
    spectrum *= taper**4
    signal = np.fft.irfft(spectrum, n).astype(np.float32)
    signal /= np.max(np.abs(signal))
    return np.tile(signal, (2, 1))


def band_energy_db(audio: np.ndarray, sr: int, low: float, high: float) -> float:
    mono = audio.mean(axis=0)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sr)
    band = spectrum[(freqs >= low) & (freqs < high)]
    rms = np.sqrt(2.0 * np.sum(np.abs(band) ** 2)) / mono.size
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


@pytest.mark.parametrize("cutoff", [5000, 8000, 11000, 16000, 20000])
def test_finds_the_brick_wall(cutoff):
    found = bandwidth_hz(band_limited(cutoff), SR)
    assert abs(found - cutoff) <= 1250, f"expected ~{cutoff}, got {found}"


@pytest.mark.parametrize("cutoff", [8000, 12000, 16000])
def test_finds_a_smeared_rolloff(cutoff):
    """The real-world case: detection must not need a perfectly sharp edge."""
    found = bandwidth_hz(smeared(cutoff), SR)
    assert abs(found - cutoff) <= 2000, f"expected ~{cutoff}, got {found}"


def test_reports_the_knee_not_the_foot():
    """The knee (where the drop starts) is what A2SB must be told. Handed the
    foot, the model reads the taper as natural spectrum and does nothing."""
    found = bandwidth_hz(smeared(12000, width_hz=1500), SR)
    assert found <= 12600, f"knee reported above the rolloff start: {found}"


def test_full_band_audio_reports_nyquist():
    """No cliff means nothing to extend; must not invent a low cutoff."""
    assert bandwidth_hz(band_limited(22050), SR) >= 19000


def test_not_fooled_by_bass_heavy_energy_distribution():
    """A 99% energy rolloff returns ~2 kHz for music regardless of bandwidth,
    because that is where the energy is. That is the upstream helper's bug."""
    signal = band_limited(16000)
    t = np.arange(signal.shape[1]) / SR
    signal = signal + 8.0 * np.sin(2 * np.pi * 60 * t)
    assert bandwidth_hz(signal, SR) > 12000


def test_detection_is_stable_across_excerpt_lengths():
    long_signal = smeared(16000, seconds=20.0)
    short = bandwidth_hz(long_signal[:, : 3 * SR], SR)
    full = bandwidth_hz(long_signal, SR)
    assert abs(short - full) <= 1000, f"{short} vs {full}"


def test_brickwall_removes_everything_above_cutoff():
    audio = band_limited(20000)
    cut = brickwall_lowpass(audio, SR, 4000)
    assert band_energy_db(cut, SR, 4500, 8000) < -120
    assert band_energy_db(cut, SR, 1000, 3500) == pytest.approx(
        band_energy_db(audio, SR, 1000, 3500), abs=0.1
    )


def test_brickwall_preserves_shape_and_is_detectable():
    audio = band_limited(20000)
    cut = brickwall_lowpass(audio, SR, 8000)
    assert cut.shape == audio.shape
    assert abs(bandwidth_hz(cut, SR) - 8000) <= 1250


def test_brickwall_accepts_mono_1d():
    mono = band_limited(20000)[0]
    cut = brickwall_lowpass(mono, SR, 8000)
    assert cut.shape == mono.shape

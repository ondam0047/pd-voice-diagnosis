"""
Acoustic feature extraction from a WAV file using Praat (parselmouth).

Expands the original app.py's feature set (F0/Range/dB/SPS only) with
voice-quality features (jitter, shimmer, HNR), formants, and MFCC. All
values are objective and reproducible from the audio alone.

Usage:
    from ml.feature_extraction import extract_features
    feats = extract_features("subject.wav").to_flat_dict()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError as e:
    raise ImportError(
        "parselmouth is required: pip install praat-parselmouth"
    ) from e

F0_MIN_DEFAULT = 75.0
F0_MAX_DEFAULT = 600.0


@dataclass
class AcousticFeatures:
    f0_mean_hz: float
    f0_sd_hz: float
    f0_range_hz: float
    f0_min_hz: float
    f0_max_hz: float
    jitter_local: float
    jitter_rap: float
    shimmer_local: float
    shimmer_apq5: float
    hnr_db: float
    intensity_mean_db: float
    intensity_sd_db: float
    f1_mean_hz: float
    f2_mean_hz: float
    f3_mean_hz: float
    mfcc_mean: list[float]
    mfcc_sd: list[float]
    duration_sec: float
    voiced_fraction: float

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        out: dict = {}
        for k, v in d.items():
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    out[f"{k}_{i + 1}"] = vi
            else:
                out[k] = v
        return out


def _safe(value, default=float("nan")) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def extract_features(
    wav_path: Path | str,
    f0_min: float = F0_MIN_DEFAULT,
    f0_max: float = F0_MAX_DEFAULT,
) -> AcousticFeatures:
    sound = parselmouth.Sound(str(wav_path))

    pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
    pitch_values = pitch.selected_array["frequency"]
    voiced = pitch_values[pitch_values > 0]
    if len(voiced) == 0:
        f0_mean = f0_sd = f0_range = f0_min_v = f0_max_v = float("nan")
        voiced_fraction = 0.0
    else:
        f0_mean = float(np.mean(voiced))
        f0_sd = float(np.std(voiced))
        f0_min_v = float(np.min(voiced))
        f0_max_v = float(np.max(voiced))
        f0_range = f0_max_v - f0_min_v
        voiced_fraction = float(len(voiced) / len(pitch_values))

    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    jitter_local = _safe(
        call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    )
    jitter_rap = _safe(
        call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    )
    shimmer_local = _safe(
        call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
    )
    shimmer_apq5 = _safe(
        call(
            [sound, point_process],
            "Get shimmer (apq5)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
    )
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
    hnr = _safe(call(harmonicity, "Get mean", 0, 0))

    intensity = sound.to_intensity()
    intensity_mean = _safe(call(intensity, "Get mean", 0, 0, "energy"))
    intensity_sd = _safe(call(intensity, "Get standard deviation", 0, 0))

    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50)
    n_frames = int(call(formant, "Get number of frames"))
    f1s, f2s, f3s = [], [], []
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        pf = call(pitch, "Get value at time", t, "Hertz", "Linear")
        if pf is None or np.isnan(pf) or pf == 0:
            continue
        v1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        v2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        v3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
        if v1 and not np.isnan(v1):
            f1s.append(v1)
        if v2 and not np.isnan(v2):
            f2s.append(v2)
        if v3 and not np.isnan(v3):
            f3s.append(v3)

    f1_mean = float(np.mean(f1s)) if f1s else float("nan")
    f2_mean = float(np.mean(f2s)) if f2s else float("nan")
    f3_mean = float(np.mean(f3s)) if f3s else float("nan")

    mfcc_obj = sound.to_mfcc(number_of_coefficients=13)
    mfcc_matrix = mfcc_obj.to_array()
    mfcc_used = mfcc_matrix[1:14, :]
    if mfcc_used.size == 0:
        mfcc_mean = [float("nan")] * 13
        mfcc_sd = [float("nan")] * 13
    else:
        mfcc_mean = np.mean(mfcc_used, axis=1).tolist()
        mfcc_sd = np.std(mfcc_used, axis=1).tolist()

    return AcousticFeatures(
        f0_mean_hz=f0_mean,
        f0_sd_hz=f0_sd,
        f0_range_hz=f0_range,
        f0_min_hz=f0_min_v,
        f0_max_hz=f0_max_v,
        jitter_local=jitter_local,
        jitter_rap=jitter_rap,
        shimmer_local=shimmer_local,
        shimmer_apq5=shimmer_apq5,
        hnr_db=hnr,
        intensity_mean_db=intensity_mean,
        intensity_sd_db=intensity_sd,
        f1_mean_hz=f1_mean,
        f2_mean_hz=f2_mean,
        f3_mean_hz=f3_mean,
        mfcc_mean=mfcc_mean,
        mfcc_sd=mfcc_sd,
        duration_sec=float(sound.get_total_duration()),
        voiced_fraction=voiced_fraction,
    )

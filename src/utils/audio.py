from __future__ import annotations

import math
from typing import Tuple

import torch
import torchaudio
from torch import Tensor


def load_audio(
    path: str,
    target_sample_rate: int,
    normalize: bool = True,
) -> Tuple[Tensor, int]:
    """Load audio file, resample nếu cần và chuẩn hoá biên độ."""
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    waveform = ensure_mono(waveform)
    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    return waveform, sample_rate


def ensure_mono(waveform: Tensor) -> Tensor:
    """Nếu waveform đa kênh -> chuyển về mono bằng trung bình."""
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def pad_or_trim(
    waveform: Tensor,
    target_num_samples: int,
    mode: str = "repeat",
) -> Tensor:
    """Đưa waveform về độ dài cố định."""
    current = waveform.size(-1)
    if current == target_num_samples:
        return waveform
    if current > target_num_samples:
        return waveform[..., :target_num_samples]

    diff = target_num_samples - current
    if mode == "zeros":
        padded = torch.nn.functional.pad(waveform, (0, diff))
    elif mode == "reflect":
        padded = torch.nn.functional.pad(waveform, (0, diff), mode="reflect")
    elif mode == "repeat":
        repeats = math.ceil(target_num_samples / current)
        padded = waveform.repeat(1, repeats)[..., :target_num_samples]
    else:
        raise ValueError(f"pad_mode không được hỗ trợ: {mode}")

    return padded

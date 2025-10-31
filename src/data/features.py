from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TypedDict

import librosa
import torch
import torchaudio
from torch import Tensor

from ..utils.audio import ensure_mono


MultiBranchFeatures = Dict[str, Tensor]


class SpectralConfig(TypedDict, total=False):
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    f_min: float
    f_max: Optional[float]
    power: float


class TemporalConfig(TypedDict, total=False):
    emphasis: bool
    highpass_cutoff: float


class CepstralConfig(TypedDict, total=False):
    hop_length: int
    n_bins: int
    bins_per_octave: int
    f_min: float


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    spectral: SpectralConfig = SpectralConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=128,
        f_min=20.0,
        f_max=None,
        power=2.0,
    )
    temporal: TemporalConfig = TemporalConfig(
        emphasis=True,
        highpass_cutoff=30.0,
    )
    cepstral: CepstralConfig = CepstralConfig(
        hop_length=256,
        n_bins=84,
        bins_per_octave=12,
        f_min=32.7,
    )


class MultiBranchFeatureExtractor:
    """Tạo đặc trưng cho 3 nhánh: spectral (Mel), temporal (raw), cepstral (CQT)."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        spec_cfg = config.spectral
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=spec_cfg.get("n_fft", 1024),
            hop_length=spec_cfg.get("hop_length", 256),
            win_length=spec_cfg.get("win_length", spec_cfg.get("n_fft", 1024)),
            f_min=spec_cfg.get("f_min", 20.0),
            f_max=spec_cfg.get("f_max"),
            n_mels=spec_cfg.get("n_mels", 128),
            power=spec_cfg.get("power", 2.0),
            normalized=False,
        )

        temp_cfg = config.temporal
        self.apply_pre_emphasis = temp_cfg.get("emphasis", True)
        self.highpass_cutoff = temp_cfg.get("highpass_cutoff", 30.0)

        self.cqt_cfg = config.cepstral

    def __call__(self, waveform: Tensor) -> MultiBranchFeatures:
        waveform = ensure_mono(waveform)
        mel = self._compute_mel_spectrogram(waveform)
        temporal = self._prepare_temporal_branch(waveform)
        cqt = self._compute_cqt(waveform)

        return {"spectral": mel, "temporal": temporal, "cepstral": cqt}

    def _compute_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        mel = self.mel_transform(waveform)
        mel = torch.log1p(mel)
        return mel

    def _prepare_temporal_branch(self, waveform: Tensor) -> Tensor:
        if self.apply_pre_emphasis:
            waveform = torchaudio.functional.preemphasis(waveform, 0.97)
        if self.highpass_cutoff is not None and self.highpass_cutoff > 0:
            waveform = torchaudio.functional.highpass_biquad(
                waveform,
                sample_rate=self.config.sample_rate,
                cutoff_freq=self.highpass_cutoff,
            )
        return waveform

    def _compute_cqt(self, waveform: Tensor) -> Tensor:
        y = waveform.squeeze(0).cpu().numpy()
        cqt = librosa.cqt(
            y,
            sr=self.config.sample_rate,
            hop_length=self.cqt_cfg.get("hop_length", 256),
            n_bins=self.cqt_cfg.get("n_bins", 84),
            bins_per_octave=self.cqt_cfg.get("bins_per_octave", 12),
            fmin=self.cqt_cfg.get("f_min", 32.7),
        )
        magnitude = torch.from_numpy((abs(cqt) ** 2).astype("float32"))
        magnitude = torch.log1p(magnitude)
        # Thêm chiều channel để thống nhất với cnn2d
        return magnitude.unsqueeze(0)

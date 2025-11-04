from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..data.features import MultiBranchFeatureExtractor
from ..models.multi_branch_model import MultiBranchAttentionModel
from ..utils.audio import load_audio, pad_or_trim
from .config import (
    build_data_module_config,
    build_model_config,
    load_yaml_config,
)


def load_model_for_inference(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    data_cfg = build_data_module_config(config["data"])
    model_cfg = build_model_config(config.get("model", {}))

    model = MultiBranchAttentionModel(model_cfg)
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)
    model.to(device_obj)

    state = torch.load(checkpoint_path, map_location=device_obj)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state)
    model.eval()

    feature_extractor = MultiBranchFeatureExtractor(data_cfg.feature)
    return {
        "model": model,
        "feature_extractor": feature_extractor,
        "data_config": data_cfg,
        "device": device_obj,
    }


def infer_audio(
    audio_path: str,
    model: MultiBranchAttentionModel,
    feature_extractor: MultiBranchFeatureExtractor,
    sample_rate: int,
    max_duration: float,
    pad_mode: str = "repeat",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    waveform, _ = load_audio(audio_path, sample_rate, normalize=True)
    max_samples = int(sample_rate * max_duration)
    waveform = pad_or_trim(waveform, max_samples, mode=pad_mode)

    features = feature_extractor(waveform)
    if device is None:
        target_device = next(model.parameters()).device
    else:
        target_device = device
    batched_features = {
        name: tensor.unsqueeze(0).to(target_device, non_blocking=True)
        for name, tensor in features.items()
    }

    model.eval()
    with torch.no_grad():
        outputs = model(batched_features)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)

    spoof_score = float(probs[0, 1].item())
    prediction_idx = int(torch.argmax(probs, dim=-1).item())
    prediction_label = "spoof" if prediction_idx == 1 else "bonafide"
    return {
        "probabilities": probs.squeeze(0).cpu().tolist(),
        "spoof_score": spoof_score,
        "prediction_index": prediction_idx,
        "prediction_label": prediction_label,
    }


def run_inference(
    audio_path: str,
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_yaml_config(config_path)
    artifacts = load_model_for_inference(config, checkpoint_path, device=device)
    data_cfg = artifacts["data_config"]
    return infer_audio(
        audio_path,
        artifacts["model"],
        artifacts["feature_extractor"],
        sample_rate=data_cfg.sample_rate,
        max_duration=data_cfg.max_duration,
        pad_mode=data_cfg.pad_mode,
        device=artifacts["device"],
    )

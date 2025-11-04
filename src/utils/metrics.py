from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


DEFAULT_TDCF_PARAMS = {
    "P_tar": 0.9802,
    "P_non": 0.0091,
    "P_spoof": 0.0107,
    "C_miss_asv": 1.0,
    "C_fa_asv": 10.0,
    "C_miss_cm": 1.0,
    "C_fa_cm": 10.0,
    "P_miss_asv": 0.05,
    "P_fa_asv": 0.01,
    "P_fa_asv_spoof": 0.30,
}


def compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel()


def compute_eer(scores: Tensor, labels: Tensor) -> float:
    """Tính Equal Error Rate (EER)."""
    labels_np = labels.detach().cpu().numpy().astype(np.int32)
    scores_np = scores.detach().cpu().numpy()

    order = np.argsort(scores_np)[::-1]
    sorted_labels = labels_np[order]

    positives = sorted_labels.sum()
    negatives = len(sorted_labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    false_accepts = 0
    false_rejects = positives
    min_gap = 1.0
    eer = 1.0

    for label in sorted_labels:
        if label == 1:
            false_rejects -= 1
        else:
            false_accepts += 1

        far = false_accepts / negatives
        frr = false_rejects / positives
        gap = abs(far - frr)
        if gap < min_gap:
            min_gap = gap
            eer = (far + frr) / 2.0
    return float(eer)


def compute_tdcf(scores: Tensor, labels: Tensor, params: Optional[Dict[str, float]] = None) -> float:
    labels_np = labels.detach().cpu().numpy().astype(np.int32)
    scores_np = scores.detach().cpu().numpy()

    bona_scores = scores_np[labels_np == 0]
    spoof_scores = scores_np[labels_np == 1]
    if bona_scores.size == 0 or spoof_scores.size == 0:
        return 0.0

    params = {**DEFAULT_TDCF_PARAMS, **(params or {})}

    c_miss_asv = params["C_miss_asv"]
    c_fa_asv = params["C_fa_asv"]
    c_miss_cm = params["C_miss_cm"]
    c_fa_cm = params["C_fa_cm"]
    p_tar = params["P_tar"]
    p_non = params["P_non"]
    p_spoof = params["P_spoof"]
    p_miss_asv = params["P_miss_asv"]
    p_fa_asv = params["P_fa_asv"]
    p_fa_asv_spoof = params["P_fa_asv_spoof"]

    thresholds = np.concatenate(([-np.inf], np.sort(scores_np), [np.inf]))
    c_default = min(c_miss_asv * p_tar, c_fa_asv * p_non)
    if c_default <= 0:
        c_default = 1.0

    asv_term = c_miss_asv * p_tar * p_miss_asv + c_fa_asv * p_non * p_fa_asv

    tdcf_values = []
    for tau in thresholds:
        p_miss_cm = float(np.mean(bona_scores >= tau))
        p_fa_cm = float(np.mean(spoof_scores < tau))
        cm_term = (
            c_miss_cm * p_tar * (1.0 - p_miss_asv) * p_miss_cm
            + c_fa_cm * p_spoof * (1.0 - p_fa_asv_spoof) * p_fa_cm
        )
        tdcf_values.append((asv_term + cm_term) / c_default)

    return float(np.min(tdcf_values))


def aggregate_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    spoof_scores = probs[:, 1]
    accuracy = compute_accuracy(logits, labels)
    eer = compute_eer(spoof_scores, labels)
    tdcf = compute_tdcf(spoof_scores, labels)
    return {"accuracy": accuracy, "eer": eer, "t_dcf": tdcf}

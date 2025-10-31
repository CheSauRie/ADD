from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


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


def aggregate_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    spoof_scores = probs[:, 1]
    accuracy = compute_accuracy(logits, labels)
    eer = compute_eer(spoof_scores, labels)
    return {"accuracy": accuracy, "eer": eer}

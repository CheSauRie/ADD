from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SpectralBranch(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv2d_block(in_channels, 32, dropout=0.1),
            nn.MaxPool2d((2, 2)),
            conv2d_block(32, 64, dropout=0.15),
            nn.MaxPool2d((2, 2)),
            conv2d_block(64, 128, dropout=0.2),
            conv2d_block(128, 128, dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class TemporalBranch(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            conv1d_block(in_channels, 32, kernel_size=11, stride=2, padding=5, dropout=0.1),
            conv1d_block(32, 64, kernel_size=9, stride=2, padding=4, dropout=0.1),
            conv1d_block(64, 128, kernel_size=7, stride=2, padding=3, dropout=0.15),
            conv1d_block(128, 128, kernel_size=5, stride=1, padding=2, dropout=0.15),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stack(x)
        x = self.temporal_pool(x)
        x = self.proj(x)
        return x


class CepstralBranch(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv2d_block(in_channels, 32, kernel_size=(3, 5), padding=(1, 2), dropout=0.1),
            nn.MaxPool2d((2, 2)),
            conv2d_block(32, 64, kernel_size=(3, 5), padding=(1, 2), dropout=0.15),
            nn.MaxPool2d((2, 2)),
            conv2d_block(64, 128, kernel_size=(3, 3), padding=(1, 1), dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return x


class AttentionFusion(nn.Module):
    """Self-attention đơn giản trên các embedding nhánh."""

    def __init__(self, embed_dim: int, attn_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branch_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            branch_embeddings: Tensor [batch, num_branches, embed_dim]
        Returns:
            fused: Tensor [batch, embed_dim]
            attn_weights: Tensor [batch, num_branches]
        """
        attn_hidden = torch.tanh(self.proj(branch_embeddings))
        scores = self.score(attn_hidden).squeeze(-1)  # [batch, num_branches]
        weights = torch.softmax(scores, dim=-1)
        branch_embeddings = self.dropout(branch_embeddings)
        fused = torch.sum(branch_embeddings * weights.unsqueeze(-1), dim=1)
        return fused, weights


@dataclass
class MultiBranchModelConfig:
    embed_dim: int = 256
    attn_dim: int = 128
    num_classes: int = 2
    classifier_hidden: int = 128
    dropout: float = 0.3


class MultiBranchAttentionModel(nn.Module):
    """Kiến trúc đa nhánh với attention fusion."""

    def __init__(self, config: MultiBranchModelConfig) -> None:
        super().__init__()
        self.config = config
        self.branches = nn.ModuleDict(
            {
                "spectral": SpectralBranch(in_channels=1, hidden_dim=config.embed_dim),
                "temporal": TemporalBranch(in_channels=1, hidden_dim=config.embed_dim),
                "cepstral": CepstralBranch(in_channels=1, hidden_dim=config.embed_dim),
            }
        )
        self.fusion = AttentionFusion(config.embed_dim, config.attn_dim, dropout=config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.num_classes),
        )

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        branch_outputs = []
        attn_order = []
        for branch_name, module in self.branches.items():
            if branch_name not in features:
                raise KeyError(f"Thiếu nhánh {branch_name} trong input features.")
            branch_out = module(features[branch_name])
            branch_outputs.append(branch_out.unsqueeze(1))
            attn_order.append(branch_name)

        branch_stack = torch.cat(branch_outputs, dim=1)  # [B, num_branches, embed_dim]
        fused, weights = self.fusion(branch_stack)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused": fused,
            "attention_weights": weights,
            "branch_embeddings": branch_stack,
            "branch_order": attn_order,
        }

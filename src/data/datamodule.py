from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import DataLoader

from .asvspoof_dataset import ASVspoofLADataset, collate_fn
from .features import FeatureConfig, MultiBranchFeatureExtractor


@dataclass
class PartitionConfig:
    partition: str
    protocol_file: Optional[str] = None
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False


@dataclass
class DataModuleConfig:
    data_root: str
    sample_rate: int = 16000
    max_duration: float = 6.0
    pad_mode: str = "repeat"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    train: Optional[PartitionConfig] = None
    valid: Optional[PartitionConfig] = None
    test: Optional[PartitionConfig] = None
    preload_waveforms: bool = False


class ASVspoofDataModule:
    """Tạo DataLoader cho train/dev/test."""

    def __init__(self, config: DataModuleConfig) -> None:
        if config.train is None or config.valid is None:
            raise ValueError("Cần cấu hình train và valid partitions.")
        self.config = config
        self.feature_extractor = MultiBranchFeatureExtractor(config.feature)
        self._datasets = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """Khởi tạo dataset theo stage."""
        if stage in (None, "fit"):
            self._datasets["train"] = self._build_dataset(self.config.train)
            self._datasets["valid"] = self._build_dataset(self.config.valid)
        if stage in (None, "test") and self.config.test is not None:
            self._datasets["test"] = self._build_dataset(self.config.test)

    def _build_dataset(self, part_cfg: PartitionConfig) -> ASVspoofLADataset:
        return ASVspoofLADataset(
            data_root=self.config.data_root,
            partition=part_cfg.partition,
            protocol_file=part_cfg.protocol_file,
            feature_extractor=self.feature_extractor,
            sample_rate=self.config.sample_rate,
            max_duration=self.config.max_duration,
            pad_mode=self.config.pad_mode,
            preload_waveforms=self.config.preload_waveforms,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self._datasets["train"], self.config.train)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self._datasets["valid"], self.config.valid)

    def test_dataloader(self) -> DataLoader:
        if "test" not in self._datasets:
            raise RuntimeError("Chưa cấu hình test dataset.")
        return self._build_loader(self._datasets["test"], self.config.test)

    def _build_loader(
        self, dataset: ASVspoofLADataset, part_cfg: PartitionConfig
    ) -> DataLoader:
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=part_cfg.batch_size,
            shuffle=part_cfg.shuffle,
            drop_last=part_cfg.drop_last,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(**loader_kwargs)

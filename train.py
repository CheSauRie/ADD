from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
import yaml

from src.data.datamodule import (
    ASVspoofDataModule,
    DataModuleConfig,
    PartitionConfig,
)
from src.data.features import FeatureConfig
from src.models.multi_branch_model import (
    MultiBranchAttentionModel,
    MultiBranchModelConfig,
)
from src.training.engine import (
    Trainer,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_feature_config(cfg: Dict[str, Any]) -> FeatureConfig:
    base = FeatureConfig()
    spectral = {**base.spectral, **cfg.get("spectral", {})}
    temporal = {**base.temporal, **cfg.get("temporal", {})}
    cepstral = {**base.cepstral, **cfg.get("cepstral", {})}
    return FeatureConfig(
        sample_rate=cfg.get("sample_rate", base.sample_rate),
        spectral=spectral,
        temporal=temporal,
        cepstral=cepstral,
    )


def build_partition_config(cfg: Dict[str, Any]) -> PartitionConfig:
    return PartitionConfig(
        partition=cfg["partition"],
        protocol_file=cfg.get("protocol_file"),
        batch_size=cfg.get("batch_size", 32),
        shuffle=cfg.get("shuffle", True),
        drop_last=cfg.get("drop_last", False),
    )


def build_data_module_config(cfg: Dict[str, Any]) -> DataModuleConfig:
    feature_cfg = build_feature_config(cfg.get("feature", {}))

    train_cfg = build_partition_config(cfg["train"])
    valid_cfg = build_partition_config(cfg["valid"])
    test_cfg = (
        build_partition_config(cfg["test"]) if cfg.get("test") is not None else None
    )

    return DataModuleConfig(
        data_root=cfg["data_root"],
        sample_rate=cfg.get("sample_rate", feature_cfg.sample_rate),
        max_duration=cfg.get("max_duration", 6.0),
        pad_mode=cfg.get("pad_mode", "repeat"),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        feature=feature_cfg,
        train=train_cfg,
        valid=valid_cfg,
        test=test_cfg,
        preload_waveforms=cfg.get("preload_waveforms", False),
    )


def build_model_config(cfg: Dict[str, Any]) -> MultiBranchModelConfig:
    base = MultiBranchModelConfig()
    return MultiBranchModelConfig(
        embed_dim=cfg.get("embed_dim", base.embed_dim),
        attn_dim=cfg.get("attn_dim", base.attn_dim),
        num_classes=cfg.get("num_classes", base.num_classes),
        classifier_hidden=cfg.get("classifier_hidden", base.classifier_hidden),
        dropout=cfg.get("dropout", base.dropout),
    )


def build_training_config(cfg: Dict[str, Any]) -> TrainingConfig:
    base = TrainingConfig()
    return TrainingConfig(
        epochs=cfg.get("epochs", base.epochs),
        device=cfg.get("device", base.device),
        log_interval=cfg.get("log_interval", base.log_interval),
        grad_clip=cfg.get("grad_clip", base.grad_clip),
        mixed_precision=cfg.get("mixed_precision", base.mixed_precision),
        checkpoint_dir=cfg.get("checkpoint_dir", base.checkpoint_dir),
        best_metric=cfg.get("best_metric", base.best_metric),
        patience=cfg.get("patience", base.patience),
        resume_from=cfg.get("resume_from", base.resume_from),
        save_every=cfg.get("save_every", base.save_every),
        evaluate_on_test=cfg.get("evaluate_on_test", base.evaluate_on_test),
    )


def build_optimizer_config(cfg: Dict[str, Any]) -> OptimizerConfig:
    base = OptimizerConfig()
    return OptimizerConfig(
        lr=cfg.get("lr", base.lr),
        weight_decay=cfg.get("weight_decay", base.weight_decay),
        betas=tuple(cfg.get("betas", base.betas)),
        eps=cfg.get("eps", base.eps),
    )


def build_scheduler_config(cfg: Dict[str, Any], total_epochs: int) -> SchedulerConfig:
    base = SchedulerConfig()
    use_cosine = cfg.get("use_cosine", base.use_cosine)
    min_lr = cfg.get("min_lr", base.min_lr)
    t_max = cfg.get("t_max", total_epochs if base.t_max is None else base.t_max)
    return SchedulerConfig(use_cosine=use_cosine, min_lr=min_lr, t_max=t_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình đa nhánh ASVspoof2019 LA")
    parser.add_argument(
        "--config",
        required=True,
        help="Đường dẫn tới file YAML cấu hình.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ghi đè thiết bị (cpu/cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    data_cfg = build_data_module_config(config["data"])
    model_cfg = build_model_config(config.get("model", {}))
    training_cfg = build_training_config(config.get("training", {}))
    if args.device is not None:
        training_cfg.device = args.device

    optimizer_cfg = build_optimizer_config(config.get("optimizer", {}))
    scheduler_cfg = build_scheduler_config(
        config.get("scheduler", {}),
        total_epochs=training_cfg.epochs,
    )

    datamodule = ASVspoofDataModule(data_cfg)
    model = MultiBranchAttentionModel(model_cfg)
    trainer = Trainer(model, training_cfg, optimizer_cfg, scheduler_cfg)

    trainer.fit(datamodule)
    print("===== Lịch sử huấn luyện =====")
    for record in trainer.train_config.history:
        epoch = record["epoch"]
        metrics_str = ", ".join(
            f"{k}={v:.4f}"
            for k, v in record.items()
            if k != "epoch"
        )
        print(f"Epoch {epoch}: {metrics_str}")

    if training_cfg.evaluate_on_test and data_cfg.test is not None:
        metrics = trainer.evaluate(datamodule)
        print("===== Kết quả Test =====")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

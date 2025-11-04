from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.datamodule import ASVspoofDataModule
from src.models.multi_branch_model import MultiBranchAttentionModel
from src.training.engine import Trainer
from src.utils.config import (
    build_data_module_config,
    build_model_config,
    build_optimizer_config,
    build_scheduler_config,
    build_training_config,
    load_yaml_config,
)


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


def plot_training_curves(history: Dict[str, List[Dict[str, float]]], output_path: str) -> None:
    if not history or not history.get("train"):
        return

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    train_history = history["train"]
    valid_history = history.get("valid", [])
    if not train_history:
        return

    metrics = list(train_history[0].keys())
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]

    epochs = list(range(1, len(train_history) + 1))
    for ax, metric in zip(axes, metrics):
        train_values = [record.get(metric) for record in train_history]
        val_values = [record.get(metric) for record in valid_history]
        ax.plot(epochs, train_values, label="Train", marker="o")
        if valid_history:
            ax.plot(epochs, val_values, label="Validation", marker="s")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_trained_model(trainer: Trainer, output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    model_state = trainer.model.state_dict()
    torch.save(model_state, output_path)
    print(f"[Trainer] Đã lưu trọng số mô hình tại {output_path}")


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

    history = trainer.fit(datamodule)

    plot_path = os.path.join(training_cfg.checkpoint_dir, "training_curves.png")
    plot_training_curves(history, plot_path)

    best_checkpoint = os.path.join(training_cfg.checkpoint_dir, "checkpoint_best.pt")
    if os.path.exists(best_checkpoint):
        trainer.load_checkpoint(best_checkpoint)
    else:
        print(
            f"[Trainer] Không tìm thấy checkpoint tốt nhất tại {best_checkpoint}. "
            "Giữ nguyên trọng số cuối cùng."
        )

    print("===== Lịch sử huấn luyện =====")
    for record in trainer.train_config.history:
        epoch = record["epoch"]
        metrics_str = ", ".join(
            f"{k}={v:.4f}"
            for k, v in record.items()
            if k != "epoch"
        )
        print(f"Epoch {epoch}: {metrics_str}")

    export_path = training_cfg.model_output_path or os.path.join(
        training_cfg.checkpoint_dir, "multibranch_model.pt"
    )
    export_trained_model(trainer, export_path)

    if training_cfg.evaluate_on_test and data_cfg.test is not None:
        metrics = trainer.evaluate(datamodule)
        print("===== Kết quả Test =====")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

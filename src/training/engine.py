from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.metrics import aggregate_metrics


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    use_cosine: bool = True
    min_lr: float = 1e-6
    t_max: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int = 50
    device: Optional[str] = None
    log_interval: int = 20
    grad_clip: float = 5.0
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    best_metric: str = "eer"
    patience: int = 10
    resume_from: Optional[str] = None
    save_every: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    evaluate_on_test: bool = False
    model_output_path: Optional[str] = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainingConfig,
        optim_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

        device_str = (
            train_config.device
            if train_config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(device_str)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda" and train_config.mixed_precision)
        )
        self.best_metric_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

    def fit(self, datamodule) -> Dict[str, List[Dict[str, float]]]:
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.val_dataloader()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optim_config.lr,
            weight_decay=self.optim_config.weight_decay,
            betas=self.optim_config.betas,
            eps=self.optim_config.eps,
        )

        scheduler = self._build_scheduler(optimizer)

        os.makedirs(self.train_config.checkpoint_dir, exist_ok=True)

        history = {"train": [], "valid": []}
        patience_counter = 0

        for epoch in range(1, self.train_config.epochs + 1):
            train_metrics = self._run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train=True,
            )
            valid_metrics = self._run_epoch(
                loader=valid_loader,
                optimizer=None,
                scheduler=None,
                epoch=epoch,
                train=False,
            )

            history["train"].append(train_metrics)
            history["valid"].append(valid_metrics)
            self.train_config.history.append(
                {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in valid_metrics.items()}}
            )

            current_metric = valid_metrics.get(self.train_config.best_metric)
            if current_metric is None:
                raise KeyError(
                    f"Không tìm thấy metric {self.train_config.best_metric} trong valid metrics: {valid_metrics}"
                )

            if self._is_better(current_metric):
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(optimizer, epoch, best=True)
            else:
                patience_counter += 1

            if self.train_config.save_every > 0 and epoch % self.train_config.save_every == 0:
                self._save_checkpoint(optimizer, epoch, best=False)

            if scheduler is not None and getattr(scheduler, "step", None) is not None:
                scheduler.step()

            if patience_counter >= self.train_config.patience:
                print(f"[Trainer] Early stopping ở epoch {epoch}.")
                break

        return history

    def evaluate(self, datamodule) -> Dict[str, float]:
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        metrics = self._run_epoch(loader=test_loader, optimizer=None, scheduler=None, epoch=0, train=False)
        return metrics

    def _run_epoch(
        self,
        loader,
        optimizer,
        scheduler,
        epoch: int,
        train: bool,
    ) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []

        for step, batch in enumerate(loader, start=1):
            features = {
                name: tensor.to(self.device, non_blocking=True)
                for name, tensor in batch["features"].items()
            }
            labels = batch["labels"].to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    outputs = self.model(features)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    if self.train_config.grad_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.train_config.grad_clip
                        )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if train and self.train_config.log_interval and step % self.train_config.log_interval == 0:
                current_loss = total_loss / total_samples
                print(
                    f"[Epoch {epoch}] Step {step}/{len(loader)} "
                    f"Loss: {current_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = total_loss / max(total_samples, 1)
        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        metrics = aggregate_metrics(logits_tensor, labels_tensor)
        metrics["loss"] = avg_loss
        return metrics

    def _is_better(self, value: float) -> bool:
        if self.best_metric_value is None:
            return True
        if self.train_config.best_metric in {"loss", "eer"}:
            return value < self.best_metric_value
        return value > self.best_metric_value

    def _save_checkpoint(self, optimizer, epoch: int, best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "best_metric": self.best_metric_value,
            "best_epoch": self.best_epoch,
        }
        suffix = "best" if best else f"epoch_{epoch:03d}"
        path = os.path.join(self.train_config.checkpoint_dir, f"checkpoint_{suffix}.pt")
        torch.save(state, path)
        tag = "BEST" if best else "SNAPSHOT"
        print(f"[Trainer] Đã lưu checkpoint ({tag}) tại {path}")

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
        state = torch.load(path, map_location=self.device)
        model_state = state.get("model_state_dict", state)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.best_metric_value = state.get("best_metric", self.best_metric_value)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        print(f"[Trainer] Đã tải checkpoint từ {path}")

    def _build_scheduler(self, optimizer):
        if not self.scheduler_config.use_cosine:
            return None
        t_max = (
            self.scheduler_config.t_max
            if self.scheduler_config.t_max is not None
            else self.train_config.epochs
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=self.scheduler_config.min_lr,
        )

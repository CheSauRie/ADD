from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .features import MultiBranchFeatureExtractor, MultiBranchFeatures
from ..utils.audio import load_audio, pad_or_trim


LA_LABELS = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class DatasetSpec:
    partition_dir_template: str
    protocol_patterns: Tuple[str, ...]
    audio_subdirs: Tuple[str, ...] = ("", "flac", "wav")
    audio_extensions: Tuple[str, ...] = (".flac", ".wav")
    protocol_dir_templates: Tuple[str, ...] = (
        "{partition_dir}/protocol",
        "{partition_dir}",
    )


DEFAULT_DATASET_VARIANT = "ASVspoof2019_LA"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "ASVspoof2019_LA": DatasetSpec(
        partition_dir_template="ASVspoof2019_LA_{partition}",
        protocol_patterns=(
            "ASVspoof2019.LA.cm.{partition}.trn.txt",
            "ASVspoof2019.LA.cm.{partition}.trl.txt",
            "ASVspoof2019.LA.cm.{partition}.txt",
        ),
        audio_subdirs=("", "flac", "wav"),
        audio_extensions=(".flac", ".wav"),
        protocol_dir_templates=(
            "ASVspoof2019_LA_cm_protocols",
            "{partition_dir}/protocol",
            "{partition_dir}",
        ),
    ),
    "ASVspoof2019_PA": DatasetSpec(
        partition_dir_template="ASVspoof2019_PA_{partition}",
        protocol_patterns=(
            "ASVspoof2019.PA.cm.{partition}.trn.txt",
            "ASVspoof2019.PA.cm.{partition}.trl.txt",
            "ASVspoof2019.PA.cm.{partition}.txt",
        ),
        audio_subdirs=("", "wav", "flac"),
        audio_extensions=(".wav", ".flac"),
        protocol_dir_templates=(
            "ASVspoof2019_PA_cm_protocols",
            "{partition_dir}/protocol",
            "{partition_dir}",
        ),
    ),
    "ASVspoof5": DatasetSpec(
        partition_dir_template="ASVspoof5_{partition}",
        protocol_patterns=(
            "ASVspoof5.cm.{partition}.txt",
            "ASVspoof5.{partition}.cm.txt",
            "ASVspoof5.{partition}.txt",
        ),
        audio_subdirs=("", "wav", "flac"),
        audio_extensions=(".wav", ".flac"),
    ),
}


@dataclass
class ASVExample:
    """Metadata container for a single ASVspoof sample."""

    utt_id: str
    speaker_id: str
    path: str
    label: int
    system_id: Optional[str] = None
    attack_type: Optional[str] = None


class ASVspoofLADataset(Dataset):
    """
    PyTorch dataset for ASVspoof2019 LA partition.

    Expects the dataset on disk with the canonical structure:

        root/
          ASVspoof2019_LA_<partition>/
            flac/
              <utt_id>.flac
            protocol/
              ASVspoof2019.LA.cm.<partition>.trn.txt

    You can override locations by providing explicit paths.
    """

    def __init__(
        self,
        data_root: str,
        partition: str,
        feature_extractor: MultiBranchFeatureExtractor,
        protocol_file: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 6.0,
        pad_mode: str = "repeat",
        preload_waveforms: bool = False,
        dataset_variant: str = DEFAULT_DATASET_VARIANT,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.partition = partition
        self.sample_rate = sample_rate
        self.max_num_samples = int(sample_rate * max_duration)
        self.pad_mode = pad_mode
        self.feature_extractor = feature_extractor
        self.preload_waveforms = preload_waveforms
        if dataset_variant not in DATASET_SPECS:
            raise ValueError(
                "Unsupported dataset_variant '{}'. Available: {}".format(
                    dataset_variant, ", ".join(DATASET_SPECS.keys())
                )
            )
        self.dataset_variant = dataset_variant
        self.dataset_spec = DATASET_SPECS[dataset_variant]

        if protocol_file is None:
            protocol_file = self._infer_protocol_path()

        self.protocol_file = protocol_file
        self.examples = self._load_metadata()

        if self.preload_waveforms:
            self._waveform_cache: Dict[str, Tensor] = {}
            for example in self.examples:
                waveform, _ = load_audio(
                    example.path, self.sample_rate, normalize=True
                )
                waveform = pad_or_trim(
                    waveform, self.max_num_samples, mode=self.pad_mode
                )
                self._waveform_cache[example.utt_id] = waveform
        else:
            self._waveform_cache = {}

    def _infer_protocol_path(self) -> str:
        partition_dir = self.dataset_spec.partition_dir_template.format(
            partition=self.partition
        )

        protocol_dirs: List[str] = []
        for template in self.dataset_spec.protocol_dir_templates:
            protocol_dirs.append(
                os.path.join(
                    self.data_root,
                    template.format(
                        partition=self.partition, partition_dir=partition_dir
                    ),
                )
            )

        # Backward compatibility: cũng tìm trong thư mục protocol nội bộ nếu có.
        protocol_dirs.extend(
            [
                os.path.join(self.data_root, partition_dir, "protocol"),
                os.path.join(self.data_root, partition_dir),
            ]
        )

        seen = set()
        protocol_dirs = [
            path for path in protocol_dirs if not (path in seen or seen.add(path))
        ]

        candidates: List[str] = []
        for proto_dir in protocol_dirs:
            for pattern in self.dataset_spec.protocol_patterns:
                pattern_path = pattern.format(partition=self.partition)
                candidates.append(os.path.join(proto_dir, pattern_path))
                # Một số bộ dữ liệu bỏ hậu tố .trn/.trl
                if pattern_path.endswith(".trn.txt"):
                    candidates.append(
                        os.path.join(proto_dir, pattern_path.replace(".trn", ""))
                    )
                if pattern_path.endswith(".trl.txt"):
                    candidates.append(
                        os.path.join(proto_dir, pattern_path.replace(".trl", ""))
                    )

        existing = [path for path in candidates if os.path.exists(path)]
        if not existing:
            raise FileNotFoundError(
                "Không tìm thấy protocol cho partition={} trong {}. Hãy cung cấp protocol_file thủ công.".format(
                    self.partition, ", ".join(protocol_dirs)
                )
            )
        return existing[0]

    def _load_metadata(self) -> List[ASVExample]:
        examples: List[ASVExample] = []
        with open(self.protocol_file, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=" ")
            for row in reader:
                tokens = [tok for tok in row if tok]
                if not tokens:
                    continue
                if len(tokens) < 3:
                    raise ValueError(
                        f"Không thể parse dòng protocol: {tokens}"
                    )

                speaker_id, utt_id = tokens[:2]
                label_token = tokens[-1].lower()

                middle_tokens = tokens[2:-1]
                system_id: Optional[str] = None
                attack_type: Optional[str] = None
                if middle_tokens:
                    system_candidate = middle_tokens[-1]
                    if system_candidate != "-":
                        system_id = system_candidate
                    if len(middle_tokens) >= 2:
                        attack_candidate = middle_tokens[0]
                        if attack_candidate != "-":
                            attack_type = attack_candidate

                if label_token not in LA_LABELS:
                    raise ValueError(f"Nhãn không hợp lệ: {label_token}")

                partition_dir = self.dataset_spec.partition_dir_template.format(
                    partition=self.partition
                )
                audio_path = None
                for subdir in self.dataset_spec.audio_subdirs:
                    audio_dir = os.path.join(self.data_root, partition_dir, subdir)
                    for ext in self.dataset_spec.audio_extensions:
                        candidate = os.path.join(audio_dir, f"{utt_id}{ext}")
                        if os.path.exists(candidate):
                            audio_path = candidate
                            break
                    if audio_path is not None:
                        break
                if audio_path is None:
                    raise FileNotFoundError(
                        f"Không tìm thấy file audio cho {utt_id} trong {partition_dir}"
                    )

                examples.append(
                    ASVExample(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        path=audio_path,
                        label=LA_LABELS[label_token],
                        system_id=system_id,
                        attack_type=attack_type,
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        example = self.examples[index]
        if example.utt_id in self._waveform_cache:
            waveform = self._waveform_cache[example.utt_id]
        else:
            waveform, _ = load_audio(
                example.path, self.sample_rate, normalize=True
            )
            waveform = pad_or_trim(
                waveform, self.max_num_samples, mode=self.pad_mode
            )

        features = self.feature_extractor(waveform)
        sample = {
            "utt_id": example.utt_id,
            "speaker_id": example.speaker_id,
            "label": torch.tensor(example.label, dtype=torch.long),
            "features": features,
        }

        metadata = {}
        if example.system_id is not None:
            metadata["system_id"] = example.system_id
        if example.attack_type is not None:
            metadata["attack_type"] = example.attack_type
        if metadata:
            sample["meta"] = metadata

        return sample


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Custom collate_fn giữ các nhánh trong tensor riêng."""
    labels = torch.stack([item["label"] for item in batch], dim=0)

    branch_tensors: Dict[str, List[Tensor]] = {}
    for item in batch:
        features: MultiBranchFeatures = item["features"]
        for branch_name, tensor in features.items():
            branch_tensors.setdefault(branch_name, []).append(tensor)

    stacked_features = {
        branch_name: torch.stack(tensors, dim=0)
        for branch_name, tensors in branch_tensors.items()
    }

    output = {
        "features": stacked_features,
        "labels": labels,
        "utt_ids": [item["utt_id"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
    }

    metas = [item.get("meta") for item in batch]
    if any(meta is not None for meta in metas):
        output["meta"] = metas

    return output

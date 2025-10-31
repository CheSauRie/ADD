from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .features import MultiBranchFeatureExtractor, MultiBranchFeatures
from ..utils.audio import load_audio, pad_or_trim


LA_LABELS = {"bonafide": 0, "spoof": 1}


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
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.partition = partition
        self.sample_rate = sample_rate
        self.max_num_samples = int(sample_rate * max_duration)
        self.pad_mode = pad_mode
        self.feature_extractor = feature_extractor
        self.preload_waveforms = preload_waveforms

        if protocol_file is None:
            proto_dir = os.path.join(
                data_root,
                f"ASVspoof2019_LA_{partition}",
                "protocol",
            )
            pattern = f"ASVspoof2019.LA.cm.{partition}.trn.txt"
            candidates = [
                os.path.join(proto_dir, pattern),
                os.path.join(proto_dir, pattern.replace(".trn", "")),
            ]
            exists = [path for path in candidates if os.path.exists(path)]
            if not exists:
                raise FileNotFoundError(
                    f"Không tìm thấy protocol cho partition={partition}. "
                    f"Hãy cung cấp protocol_file thủ công. Checked: {candidates}"
                )
            protocol_file = exists[0]

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

    def _load_metadata(self) -> List[ASVExample]:
        examples: List[ASVExample] = []
        with open(self.protocol_file, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=" ")
            for row in reader:
                tokens = [tok for tok in row if tok]
                if not tokens:
                    continue
                if len(tokens) == 4:
                    speaker_id, utt_id, system_id, label_token = tokens
                    attack_type = None
                elif len(tokens) >= 5:
                    speaker_id, utt_id, system_id, attack_type, label_token = tokens[:5]
                else:
                    raise ValueError(
                        f"Không thể parse dòng protocol: {tokens}"
                    )

                label_token = label_token.lower()
                if label_token not in LA_LABELS:
                    raise ValueError(f"Nhãn không hợp lệ: {label_token}")

                partition_dir = f"ASVspoof2019_LA_{self.partition}"
                audio_dir = os.path.join(self.data_root, partition_dir, "flac")
                audio_path = os.path.join(audio_dir, f"{utt_id}.flac")
                if not os.path.exists(audio_path):
                    # Một số dataset dùng .wav
                    wav_path = os.path.join(audio_dir, f"{utt_id}.wav")
                    if os.path.exists(wav_path):
                        audio_path = wav_path
                    else:
                        raise FileNotFoundError(
                            f"Không tìm thấy file audio cho {utt_id} tại {audio_dir}"
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

from typing import Any, Literal

import torch
from torch.utils.data import Dataset


class BetDataset(Dataset):
    def __init__(
        self,
        data: dict[str, list],
        dataset_type: Literal["train", "test", "val"],
    ) -> None:
        self._data = data
        self.dataset_type = dataset_type

    def __getitem__(self, idx: int) -> Any:
        features = self._data["features"][idx]
        true_label = self._data["targets"][idx]

        observation = {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(true_label, dtype=torch.float32),
        }

        return observation

    def __len__(self) -> int:
        return len(self._data["targets"])

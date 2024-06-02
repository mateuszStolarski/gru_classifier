import json
from pathlib import Path

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.dataset.bet import BetDataset
from src.settings.data import DataSettings
from src.utils.data_helper import collate_fn, create_splitted_sequences


class BetDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        data_settings: DataSettings,
        collate_fn: callable = collate_fn,
        seed: int = 2024,
        num_workers: int = 2,
        real_batch_size: int = 1,
    ) -> None:
        super().__init__()
        data_path.mkdir(
            exist_ok=True,
            parents=True,
        )
        self._data_path = data_path
        self._data_settings = data_settings
        self._seed = seed
        self._batch_size = real_batch_size
        self._num_workers = num_workers
        self._collate_fn = collate_fn

        self.train, self.test, self.val = None, None, None

    def prepare_data(self) -> None:
        if not all(
            [
                (self._data_path / f"train.json").exists(),
                (self._data_path / f"val.json").exists(),
                (self._data_path / f"test.json").exists(),
            ]
        ):
            sequences = create_splitted_sequences(self._data_settings)
            for split, data in sequences.items():
                save_path = self._data_path / f"{split}.json"
                save_path.touch(exist_ok=True)
                with open(
                    file=save_path,
                    mode="w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        obj=data,
                        fp=file,
                        indent=4,
                    )
        return super().prepare_data()

    def setup(
        self,
        stage: str,
    ) -> None:
        match stage:
            case "fit":
                with open(
                    file=self._data_path / "train.json",
                    mode="r",
                    encoding="utf-8",
                ) as file:
                    self.train = BetDataset(
                        data=json.load(file),
                        dataset_type="train",
                    )
                with open(
                    file=self._data_path / "val.json",
                    mode="r",
                    encoding="utf-8",
                ) as file:
                    self.val = BetDataset(
                        data=json.load(file),
                        dataset_type="val",
                    )
            case "test":
                with open(
                    file=self._data_path / "test.json",
                    mode="r",
                    encoding="utf-8",
                ) as file:
                    self.test = BetDataset(
                        data=json.load(file),
                        dataset_type="test",
                    )

    def get_targets(self, stage: str) -> list[int]:
        with open(
            file=self._data_path / f"{stage}.json",
            mode="r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
            targets = data["targets"]

        return targets

    def get_data(self, stage: str) -> BetDataset:
        with open(
            file=self._data_path / f"{stage}.json",
            mode="r",
            encoding="utf-8",
        ) as file:
            data = BetDataset(
                data=json.load(file),
                dataset_type=stage,
            )

        return data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            shuffle=True,
        )

import datetime
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup

from src.wrapper import WrapperConfig


class BetWrapper(pl.LightningModule):
    def __init__(
        self,
        device: str,
        real_batch_size: int,
        learning_rate: float,
        student: torch.nn.Module,
        config: WrapperConfig,
        class_weights: list[float] = None,
    ) -> None:
        super().__init__()
        self._device = device
        self._real_batch_size = real_batch_size
        self._learning_rate = learning_rate

        self._student = student
        self._config = config

        self._class_weights = class_weights
        if class_weights:
            self._loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self._class_weights).to(self._device)
            ).to(self._device)
        else:
            self._loss_fn = torch.nn.CrossEntropyLoss().to(self._device)

        self.test_preds = {
            "trues": [],
            "preds": [],
        }

    def process(
        self,
        batch: dict[str, torch.Tensor],
        is_train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch.pop("features").to(self._device)
        labels = batch.pop("labels").to(self._device).long()

        if not is_train:
            with torch.no_grad():
                outputs = self._student(features)
        else:
            outputs = self(features)

        loss = self._loss_fn(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)

        return labels, predictions, loss

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            return self._student(features)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        _, _, loss = self.process(batch, is_train=True)

        self.log(
            name="train_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self._real_batch_size,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        _, _, loss = self.process(batch)

        self.log(
            name=f"val_loss",
            value=loss,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self._real_batch_size,
        )

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        labels, predictions, loss = self.process(batch)

        self.test_preds["trues"].extend(labels.tolist())
        self.test_preds["preds"].extend(predictions.tolist())

        self.log(
            name=f"test_loss",
            value=loss,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self._real_batch_size,
        )

        return loss

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._learning_rate,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self._config.num_warmup_steps,
            num_training_steps=self._config.max_steps,
            num_cycles=self._config.num_cycles,
            last_epoch=-1,
        )

        return [optimizer], [scheduler]

    def save_test_results(self, save_path: Path) -> None:
        save_path.mkdir(
            exist_ok=True,
            parents=True,
        )
        x = datetime.datetime.now()
        x = x.strftime("%y_%m_%d_%H_%M")
        with (save_path / f"{x}_results.json").open("w") as file:
            json.dump(self.test_preds, file)

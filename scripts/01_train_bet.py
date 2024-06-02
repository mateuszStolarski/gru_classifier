import logging
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report

import __init__
from src import DEVICE, RANDOM_STATE
from src.datamodule.bet import BetDataModule
from src.model.bet import BetClassifier
from src.settings import class_from_args
from src.settings.data import DataSettings
from src.utils.data_helper import get_class_weights
from src.utils.model_helper import get_config
from src.wrapper.bet import BetWrapper

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_STATE)
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="01_config",
)
def main(config: DictConfig):
    save_path = Path.cwd().parents[0] / config["base"]["save_path"]

    logger.info("Preparing data")
    data_module = BetDataModule(
        data_path=save_path / "train_data",
        data_settings=class_from_args(
            className=DataSettings,
            argDict=config["data"],
        ),
        real_batch_size=config["base"]["real_batch_size"],
        seed=RANDOM_STATE,
    )
    data_module.prepare_data()

    logger.info("Preparing model")
    wrapper = BetWrapper(
        device=DEVICE,
        real_batch_size=config["base"]["real_batch_size"],
        learning_rate=config["base"]["learning_rate"],
        student=BetClassifier(
            batch_size=config["base"]["real_batch_size"],
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
        ),
        config=get_config(
            len_train_dataset=len(data_module.get_data("train")),
            len_val_dataset=len(data_module.get_data("val")),
            len_test_dataset=len(data_module.get_data("test")),
            max_epochs=config["base"]["max_epoch"],
            num_warmup_steps=config["model"]["num_warmup_steps"],
            num_cycles=config["model"]["num_cycles"],
        ),
        class_weights=get_class_weights(data_module.get_targets("train")),
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        dirpath=save_path / "model",
        monitor="val_loss",
        filename="bet-lstm-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=config["base"]["max_epoch"],
        enable_checkpointing=True,
        strategy="auto",
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        log_every_n_steps=1,
        logger=TensorBoardLogger(
            save_dir=save_path.parent,
            name="tb_logs",
        ),
        callbacks=[
            callbacks.GradientAccumulationScheduler(
                scheduling={0: config["base"]["batch_size"]}
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(config["base"]["max_epoch"] / 3),
            ),
            model_checkpoint,
        ],
    )

    logger.info("Training model")
    trainer.fit(
        model=wrapper,
        datamodule=data_module,
    )
    test_result = trainer.test(
        model=wrapper,
        datamodule=data_module,
    )

    logger.info(model_checkpoint.best_model_path)
    wrapper.save_test_results(save_path=save_path / "results")

    logger.info(test_result)
    logger.info(
        classification_report(
            wrapper.test_preds["trues"],
            wrapper.test_preds["preds"],
        ),
    )


if __name__ == "__main__":
    main()

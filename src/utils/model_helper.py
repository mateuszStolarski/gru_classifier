from src.wrapper import WrapperConfig


def _get_max_steps_lenghts(
    len_train_dataset: int,
    len_test_dataset: int,
    len_val_dataset: int,
    max_epochs: int,
) -> int:
    test_size = len_train_dataset / len_test_dataset
    val_size = len_train_dataset / len_val_dataset
    return round(len_train_dataset * (1.0 - test_size - val_size)) * max_epochs


def get_config(
    len_train_dataset: int,
    len_test_dataset: int,
    len_val_dataset: int,
    max_epochs: int,
    num_warmup_steps: int,
    num_cycles: int,
) -> WrapperConfig:
    return WrapperConfig(
        num_warmup_steps=num_warmup_steps,
        max_steps=_get_max_steps_lenghts(
            len_train_dataset=len_train_dataset,
            len_val_dataset=len_val_dataset,
            len_test_dataset=len_test_dataset,
            max_epochs=max_epochs,
        ),
        num_cycles=num_cycles,
    )

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src import BET_PATH, DROP_PATH, RANDOM_STATE
from src.settings.data import DataSettings
from src.utils.target_helper import compute_target

FEATURES = [
    "stawka",
    "wygrana",
    "ilosc_zakladow",
    "kwota_wplaty",
    "wplata",
    "wyplata",
    "kwota_wyplaty",
    "saldo",
]


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "features": torch.stack([i["features"] for i in batch]),
        "labels": torch.stack([i["labels"] for i in batch]),
    }


def _get_bets() -> pd.DataFrame:
    df = pd.read_csv(
        BET_PATH,
        sep=";",
    )
    df = df.dropna()
    df = df.sort_values(by=["data"])
    df = df[df["stawka"] != "x"]
    df["stawka"] = df["stawka"].astype(float)
    df["wygrana"] = df["wygrana"].astype(float)
    df["user_id"] = df["user"]
    df = (
        df.groupby(["data", "user"])
        .agg(
            {
                "stawka": "sum",
                "wygrana": "sum",
                "user_id": "count",
            }
        )
        .reset_index()
    )
    df.rename(
        columns={"user_id": "ilosc_zakladow"},
        inplace=True,
    )

    return df


def _get_drops() -> pd.DataFrame:
    df = pd.read_csv(DROP_PATH, sep=";")
    df = df.dropna()
    df = df.sort_values(by=["data"])
    df["wplata"] = (df["czy_wplata"] == 1).astype(int)
    df["wyplata"] = (df["czy_wplata"] == 0).astype(int)
    df["kwota_wplaty"] = df["kwota"] * df["czy_wplata"]
    df["kwota_wyplaty"] = df["kwota"] * (1 - df["czy_wplata"])
    df = (
        df.groupby(["data", "user"])
        .agg(
            {
                "kwota_wplaty": "sum",
                "wplata": "sum",
                "wyplata": "sum",
                "kwota_wyplaty": "sum",
            }
        )
        .reset_index()
    )

    return df


def _get_data() -> dict[str, pd.DataFrame]:
    bet_df = _get_bets()
    drop_df = _get_drops()

    merged_df = pd.merge(
        left=bet_df,
        right=drop_df,
        on=["user", "data"],
        how="outer",
    )
    merged_df.fillna(
        value=0,
        inplace=True,
    )
    merged_df["temp"] = (
        merged_df["wygrana"] + merged_df["kwota_wplaty"] - merged_df["kwota_wyplaty"]
    )
    merged_df["saldo"] = merged_df.groupby("user")["temp"].cumsum()
    merged_df = merged_df.drop(
        columns=["temp"],
        axis=1,
    )

    return merged_df


def _get_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    return scaler.fit(df[FEATURES])


def _get_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    unique_users = df["user"].unique()

    users_train, users_temp = train_test_split(
        unique_users,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )
    users_valid, users_test = train_test_split(
        users_temp,
        test_size=float(1 / 3),
        random_state=RANDOM_STATE,
    )

    train_df = df[df["user"].isin(users_train)]
    val_df = df[df["user"].isin(users_valid)]
    test_df = df[df["user"].isin(users_test)]

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def _create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    lack_of_activiti: int,
    empty_balance: int,
    scaler: StandardScaler,
) -> dict[str, list]:
    users = np.unique(df["user"]).tolist()
    sequences = []
    targets = []

    for user in users:
        data = df[df["user"] == user]
        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i : i + sequence_length][FEATURES]
            target = compute_target(
                df=sequence,
                lack_of_activiti=lack_of_activiti,
                empty_balance=empty_balance,
            )
            sequence = scaler.transform(sequence)

            sequences.append(sequence.tolist())
            targets.append(target)

    return {
        "features": sequences,
        "targets": targets,
    }


def create_splitted_sequences(
    data_settings: DataSettings,
) -> dict[str, dict[str, list]]:
    df = _get_data()

    scaler = _get_scaler(df)
    splits = _get_splits(df)

    for key, df in splits.items():
        splits[key] = _create_sequences(
            df=df,
            sequence_length=data_settings.sequence_length,
            lack_of_activiti=data_settings.lack_of_activiti,
            empty_balance=data_settings.empty_balance,
            scaler=scaler,
        )

    return splits


def get_class_weights(targets: list[int]) -> list[float]:
    return compute_class_weight(
        class_weight="balanced",
        classes=[0, 1],
        y=targets,
    ).tolist()

import pandas as pd


def compute_target(
    df: pd.DataFrame,
    lack_of_activiti: int,
    empty_balance: int,
) -> int:
    is_active = _lack_of_activity(
        df=df,
        lack_of_activiti=lack_of_activiti,
    )
    has_money = _empty_balance(
        df=df,
        empty_balance=empty_balance,
    )

    return int(any([is_active, has_money]))


def _calculate_streak(
    df: pd.DataFrame,
    column_name: str,
) -> int:
    reversed_column = df[column_name][::-1].reset_index(drop=True)
    binary_series = reversed_column == 0

    if not binary_series.any():
        return 0
    else:
        streak_length = binary_series.idxmin()

    return streak_length


def _lack_of_activity(
    df: pd.DataFrame,
    lack_of_activiti: int,
) -> bool:
    bets_number = _calculate_streak(df, "ilosc_zakladow")
    incoms_number = _calculate_streak(df, "wplata")
    outcoms_number = _calculate_streak(df, "wyplata")

    return max([bets_number, incoms_number, outcoms_number]) >= lack_of_activiti


def _empty_balance(
    df: pd.DataFrame,
    empty_balance: int,
) -> bool:
    is_empty = df["saldo"] == 0.0
    empty_group = (is_empty != is_empty.shift()).cumsum()
    empty_streaks = is_empty.groupby(empty_group).sum()

    empty_streaks = empty_streaks[empty_streaks > 0]
    max_empty_streak = empty_streaks.max() if not empty_streaks.empty else 0

    return max_empty_streak >= empty_balance

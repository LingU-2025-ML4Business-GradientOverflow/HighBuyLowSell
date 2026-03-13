from __future__ import annotations

import pandas as pd


def build_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact baseline feature table for next-day direction prediction."""
    frame = data.copy()
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = frame.groupby("symbol", group_keys=False)

    frame["return_1d"] = grouped["close"].pct_change(1)
    frame["return_5d"] = grouped["close"].pct_change(5)
    frame["volume_change_1d"] = grouped["volume"].pct_change(1)
    frame["ma_5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    frame["ma_10"] = grouped["close"].transform(lambda s: s.rolling(10).mean())
    frame["volatility_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(5).std())

    frame["ma_gap_5"] = frame["close"] / frame["ma_5"] - 1.0
    frame["ma_gap_10"] = frame["close"] / frame["ma_10"] - 1.0

    frame["future_return_1d"] = grouped["close"].shift(-1) / frame["close"] - 1.0
    frame["target_direction"] = (frame["future_return_1d"] > 0).astype(int)

    feature_columns = [
        "return_1d",
        "return_5d",
        "volume_change_1d",
        "ma_gap_5",
        "ma_gap_10",
        "volatility_5d",
    ]

    required_columns = ["date", "symbol", "close", "future_return_1d", "target_direction"]
    frame = frame.dropna(subset=feature_columns + ["future_return_1d"]).reset_index(drop=True)
    return frame[required_columns + feature_columns]


def feature_columns() -> list[str]:
    return [
        "return_1d",
        "return_5d",
        "volume_change_1d",
        "ma_gap_5",
        "ma_gap_10",
        "volatility_5d",
    ]

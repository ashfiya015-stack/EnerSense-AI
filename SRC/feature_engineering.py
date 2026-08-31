import pandas as pd


def create_time_features(df):
    """
    Create calendar/time-based features from timestamp.
    """

    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_lag_features(
    df,
    target="energy_consumption"
):
    """
    Create lag and rolling-window features.
    """

    df = df.copy()

    df["lag_1h"] = df[target].shift(1)

    df["lag_2h"] = df[target].shift(2)

    df["lag_24h"] = df[target].shift(24)

    df["lag_168h"] = df[target].shift(168)

    df["rolling_24h"] = (
        df[target]
        .rolling(24)
        .mean()
    )

    df["rolling_168h"] = (
        df[target]
        .rolling(168)
        .mean()
    )

    return df


def prepare_model_data(df):
    """
    Create model-ready dataset by removing rows
    created with missing lag/rolling values.
    """

    df = df.copy()

    df = df.dropna().reset_index(drop=True)

    return df

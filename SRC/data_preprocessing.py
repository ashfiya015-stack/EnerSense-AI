import pandas as pd


def load_data(electricity_path, metadata_path, weather_path):
    """
    Load electricity, metadata, and weather datasets.
    """

    electricity = pd.read_csv(electricity_path)
    metadata = pd.read_csv(metadata_path)
    weather = pd.read_csv(weather_path)

    return electricity, metadata, weather


def prepare_timestamps(electricity, weather):
    """
    Convert timestamp columns to datetime format.
    """

    electricity = electricity.copy()
    weather = weather.copy()

    electricity["timestamp"] = pd.to_datetime(
        electricity["timestamp"]
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"]
    )

    return electricity, weather


def handle_missing_values(df):
    """
    Handle missing numerical values using interpolation,
    forward fill, and backward fill.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .interpolate()
        .ffill()
        .bfill()
    )

    return df


def select_building(electricity, building_id):
    """
    Select one building/meter from the electricity dataset.
    """

    electricity = electricity.copy()

    selected = electricity[
        ["timestamp", building_id]
    ].copy()

    selected = selected.rename(
        columns={
            building_id: "energy_consumption"
        }
    )

    return selected

import os
import joblib

from sklearn.ensemble import (
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_histgradient_model(
    X_train,
    y_train,
    random_state=42
):
    """
    Train the HistGradientBoosting regression model.
    """

    model = HistGradientBoostingRegressor(
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Calculate model performance metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def save_model(
    model,
    model_path
):
    """
    Save trained model using joblib.
    """

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )


def load_model(model_path):
    """
    Load a saved model.
    """

    return joblib.load(
        model_path
    )


def save_features(
    features,
    features_path
):
    """
    Save model feature list.
    """

    os.makedirs(
        os.path.dirname(features_path),
        exist_ok=True
    )

    joblib.dump(
        features,
        features_path
    )


def load_features(
    features_path
):
    """
    Load saved model feature list.
    """

    return joblib.load(
        features_path
    )

import pandas as pd


def calculate_prediction_errors(
    actual,
    predicted
):
    """
    Calculate prediction error and absolute error.
    """

    results = pd.DataFrame({
        "actual": actual,
        "predicted": predicted
    })

    results["error"] = (
        results["actual"]
        - results["predicted"]
    )

    results["absolute_error"] = (
        results["error"]
        .abs()
    )

    return results


def detect_anomalies(
    actual,
    predicted,
    threshold
):
    """
    Detect observations with prediction error
    above the specified threshold.
    """

    results = calculate_prediction_errors(
        actual,
        predicted
    )

    results["is_anomaly"] = (
        results["absolute_error"]
        > threshold
    )

    return results


def anomaly_summary(results):
    """
    Return basic anomaly statistics.
    """

    total = len(results)

    anomalies = results[
        results["is_anomaly"]
    ].sum()["is_anomaly"]

    percentage = (
        anomalies / total * 100
        if total > 0
        else 0
    )

    return {
        "total_observations": total,
        "anomalies": int(anomalies),
        "anomaly_percentage": percentage
    }

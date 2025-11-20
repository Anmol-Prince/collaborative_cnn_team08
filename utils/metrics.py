import numpy as np
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

def compute_classification_metrics(y_true, y_pred, average="macro"):
    """
    Compute common classification metrics for classification tasks.

    Args:
        y_true (list or np.array): Ground truth labels
        y_pred (list or np.array): Predicted labels
        average (str): Averaging mode for multi-class classification.
                       Options: "macro", "micro", "weighted"

    Returns:
        dict: Accuracy, F1, Precision, Recall, Confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }

    try:
        cm = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        cm = None

    metrics["confusion_matrix"] = cm

    return metrics


def save_metrics(metrics: dict, path: str):
    """
    Save a dictionary of metrics to a JSON file.

    Args:
        metrics (dict): metrics dictionary
        path (str): output JSON file path
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

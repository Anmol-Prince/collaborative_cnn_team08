import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(true_labels, pred_labels, average="macro"):
    """
    Compute standard classification metrics.
    
    Args:
        true_labels (list or array): Ground truth label indices
        pred_labels (list or array): Predicted label indices
        average (str): F1 averaging method
    
    Returns:
        dict: accuracy, f1, and confusion matrix
    """
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average=average)
    cm = confusion_matrix(true_labels, pred_labels)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }


def save_metrics(metrics_dict, file_path):
    """
    Save metrics dictionary to a JSON file.
    
    Args:
        metrics_dict (dict): Metrics dictionary
        file_path (str or Path): Output JSON path
    """
    with open(file_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {file_path}")


def load_metrics(file_path):
    """
    Load metrics dictionary from a JSON file.
    
    Args:
        file_path (str or Path)
    
    Returns:
        dict
    """
    with open(file_path, "r") as f:
        return json.load(f)


def print_metrics(metrics_dict):
    """
    Nicely print accuracy, F1, confusion matrix.
    """
    print("\n===== Metrics =====")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
    print("===================\n")

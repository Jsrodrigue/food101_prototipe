from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, metrics_list):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    results = {}
    for m in metrics_list:
        if m == "accuracy":
            results[m] = accuracy_score(y_true, y_pred)
        elif m == "precision_macro":
            results[m] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        elif m == "recall_macro":
            results[m] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        elif m == "f1_macro":
            results[m] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown metric {m}")
    return results

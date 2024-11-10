#Utility functions (e.g., metrics calculation, data transformations)

from sklearn.metrics import accuracy_score, mean_squared_error

def calculate_metrics(y_true, y_pred, task="classification"):
    if task == "classification":
        return accuracy_score(y_true, y_pred)
    elif task == "regression":
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError("Unknown task type.")


from pathlib import Path

import joblib

# Resolved from this file, so the code works no matter which folder you run it from
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "diabetes_model.pkl"


def predict_data(X):
    """
    Predict the class label and the model's confidence for each input row.
    Args:
        X (array-like): Rows of the 8 feature values, in CSV column order.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels (0 or 1).
        confidence (numpy.ndarray): Probability the model gives to its predicted class.
    """
    model = joblib.load(MODEL_PATH)
    return model.predict(X), model.predict_proba(X).max(axis=1)

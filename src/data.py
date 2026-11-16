from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"


def load_data():
    """
    Load the Pima Indians Diabetes dataset (Kaggle: uciml/pima-indians-diabetes-database).
    Returns:
        X (numpy.ndarray): The 8 feature columns, in CSV column order.
        y (numpy.ndarray): The Outcome column (1 = diabetic, 0 = not diabetic).
    """
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    return data[:, :-1], data[:, -1].astype(int)


def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test

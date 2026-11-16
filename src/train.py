from sklearn.tree import DecisionTreeClassifier
import joblib
from data import load_data, split_data
from predict import MODEL_PATH


def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        DecisionTreeClassifier: The fitted model.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, MODEL_PATH)
    return dt_classifier


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

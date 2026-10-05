import pandas as pd
from sklearn.metrics import ( accuracy_score, classification_report, confusion_matrix )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """Evaluate a trained model on the test set."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
    )
    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    return y_pred, accuracy, report, matrix
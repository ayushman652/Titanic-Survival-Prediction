from src.data_loader import load_data
from src.evaluator import evaluate_model
from src.preprocessing import build_preprocessor, split_data
from src.trainer import (
    train_logistic_regression,
    train_random_forest,
)
from src.visualizer import (
    plot_coefficients,
    plot_confusion_matrix,
    plot_feature_importances,
)


def get_feature_names(
    model,
    numerical_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Get feature names after preprocessing."""

    categorical_feature_names = (
        model.best_estimator_["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )

    return numerical_features + list(
        categorical_feature_names
    )


def main() -> None:
    """Run the Titanic survival prediction project."""

    dataframe = load_data()

    X_train, X_test, y_train, y_test = split_data(
        dataframe
    )

    preprocessor, numerical_features, categorical_features = build_preprocessor(X_train)

    print("\nTraining Random Forest...")
    random_forest_model = train_random_forest(
        preprocessor,
        X_train,
        y_train,
    )

    (
        rf_predictions,
        rf_accuracy,
        rf_report,
        rf_matrix,
    ) = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
    )

    print("\nRandom Forest Results")
    print(f"Best Parameters: {random_forest_model.best_params_}")
    print(f"Best CV Accuracy: {random_forest_model.best_score_:.4f}")
    print(f"Test Accuracy: {rf_accuracy:.4f}")
    print(rf_report)

    plot_confusion_matrix(
        rf_matrix,
        "Random Forest",
    )

    feature_names = get_feature_names(
        random_forest_model,
        numerical_features,
        categorical_features,
    )

    plot_feature_importances(
        feature_names,
        random_forest_model.best_estimator_[
            "classifier"
        ].feature_importances_,
    )

    print("\nTraining Logistic Regression...")
    logistic_regression_model = train_logistic_regression(
        preprocessor,
        X_train,
        y_train,
    )

    (
        lr_predictions,
        lr_accuracy,
        lr_report,
        lr_matrix,
    ) = evaluate_model(
        logistic_regression_model,
        X_test,
        y_test,
    )

    print("\nLogistic Regression Results")
    print(
        f"Best Parameters: "
        f"{logistic_regression_model.best_params_}"
    )
    print(
        f"Best CV Accuracy: "
        f"{logistic_regression_model.best_score_:.4f}"
    )
    print(f"Test Accuracy: {lr_accuracy:.4f}")
    print(lr_report)

    plot_confusion_matrix(
        lr_matrix,
        "Logistic Regression",
    )

    plot_coefficients(
        feature_names,
        logistic_regression_model.best_estimator_[
            "classifier"
        ].coef_[0],
    )


if __name__ == "__main__":
    main()
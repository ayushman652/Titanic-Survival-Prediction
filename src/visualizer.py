import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import OUTPUT_DIR


def plot_confusion_matrix(
    matrix,
    model_name: str,
) -> None:
    """Plot and save a confusion matrix."""

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title(f"Titanic Classification - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_feature_importances(
    feature_names: list[str],
    importances,
) -> None:
    """Plot Random Forest feature importances."""

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )

    plt.gca().invert_yaxis()
    plt.title(
        "Random Forest Feature Importances"
    )
    plt.xlabel("Importance Score")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "random_forest_feature_importances.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_coefficients(
    feature_names: list[str],
    coefficients,
) -> None:
    """Plot Logistic Regression coefficient magnitudes."""

    coefficient_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    ).sort_values(
        by="Coefficient",
        key=abs,
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        coefficient_df["Feature"],
        coefficient_df["Coefficient"].abs(),
    )

    plt.gca().invert_yaxis()
    plt.title(
        "Logistic Regression Coefficient Magnitudes"
    )
    plt.xlabel("Coefficient Magnitude")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "logistic_regression_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
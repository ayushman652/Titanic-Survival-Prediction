from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import (
    CV_SPLITS,
    LOGISTIC_REGRESSION_PARAM_GRID,
    RANDOM_FOREST_PARAM_GRID,
    RANDOM_STATE,
    SCORING,
)


def train_model(
    preprocessor,
    classifier,
    param_grid: dict,
    X_train,
    y_train,
) -> GridSearchCV:
    """Build a pipeline, tune it with GridSearchCV, and train it."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=SCORING,
        verbose=0,
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(
    preprocessor,
    X_train,
    y_train,
) -> GridSearchCV:
    """Train and tune the Random Forest classifier."""

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    return train_model(
        preprocessor,
        classifier,
        RANDOM_FOREST_PARAM_GRID,
        X_train,
        y_train,
    )


def train_logistic_regression(
    preprocessor,
    X_train,
    y_train,
) -> GridSearchCV:
    """Train and tune the Logistic Regression classifier."""

    classifier = LogisticRegression(
        random_state=RANDOM_STATE
    )

    return train_model(
        preprocessor,
        classifier,
        LOGISTIC_REGRESSION_PARAM_GRID,
        X_train,
        y_train,
    )
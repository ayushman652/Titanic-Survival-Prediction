from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


TEST_SIZE = 0.20
RANDOM_STATE = 42

CV_SPLITS = 5
SCORING = "accuracy"

FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "class",
    "who",
    "adult_male",
    "alone",
]

TARGET = "survived"

RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

LOGISTIC_REGRESSION_PARAM_GRID = {
    "classifier__solver": ["liblinear"],
    "classifier__penalty": ["l1", "l2"],
    "classifier__class_weight": [None, "balanced"],
}
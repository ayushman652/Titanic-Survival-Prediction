import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def split_data( dataframe: pd.DataFrame ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Select features and split the data into training and test sets."""

    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_preprocessor( X_train: pd.DataFrame ) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build preprocessing pipelines for numerical and categorical features."""

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numerical_features, categorical_features
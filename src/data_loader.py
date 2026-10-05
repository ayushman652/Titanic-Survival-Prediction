import pandas as pd
import seaborn as sns


def load_data() -> pd.DataFrame:
    """Load the Titanic dataset from Seaborn."""
    return sns.load_dataset("titanic")
"""Module containing custom transformers for feature engineering."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BinTransformer(BaseEstimator, TransformerMixin):
    """Transformer that bins numeric values into integer bins.

    Values outside of bins and/or NA values are returned as -1. NA values can be
    removed before conversion using the `fill_values` parameter.

    Parameters
    ----------
    bins : Dict[str, List[int]]
        Cutoffs to use when binning values. Specified as a Dict, in which the keys
        reflect the column to which the corresponding bins should be applied.
    fill_values : Dict[int]
        Fill values to use for NA's, specified per column. As such, keys should match
        the keys used for the bin parameter.

    """

    def __init__(self, bins, fill_values=None):
        self.bins = bins
        self.fill_values = fill_values or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        if copy:
            X = X.copy()

        for col, bins_ in self.bins.items():
            if col in self.fill_values:
                X[col] = X[col].fillna(self.fill_values[col])
            X[col] = pd.cut(X[col], bins_, labels=range(len(bins_[:-1]))).cat.codes

        return X


class CabinTransformer(BaseEstimator, TransformerMixin):
    """Transformer that simplifies cabin designations into categories."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        if copy:
            X = X.copy()

        cabins = X["Cabin"].fillna("N")
        X["Cabin"] = cabins.apply(lambda x: x[0])

        return X


class SelectColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformer that selects specific columns from a pandas DataFrame.

    Parameters
    ----------
    columns : List[str]
        Columns to select from the Dataframe.

    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        return X[self.columns]


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Transformer that encodes pre-defined categorical values into
       ordinal integer values.

    Parameters
    ----------
    categories : Dict[str, List[str]]
        Categories to use for each column. Specified as a Dict of lists, with column
        names in the keys and the corresponding (ordered) categorical values of that
        column in the values of the list.
    fill_value : int
        Fill value to use for NAs or unknown values.

    """
    def __init__(self, categories, fill_value=-1):
        self.categories = categories
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        if copy:
            X = X.copy()

        for col, categories in self.categories.items():
            # Map values, filling NaNs with fill_value.
            value_map = dict(zip(categories, range(len(categories))))
            encoded = X[col].map(value_map).fillna(self.fill_value)

            # Try to convert to int (should succeed if there are no NaNs).
            try:
                encoded = encoded.astype(int)
            except ValueError:
                pass

            X[col] = encoded

        return X

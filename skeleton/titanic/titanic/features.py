# -*- coding: utf-8 -*-

"""Classes/functions for performing feature engineering."""

from sklearn.base import BaseEstimator, TransformerMixin


class CabinTransformer(BaseEstimator, TransformerMixin):
    """Transformer that simplifies cabin designations into categories."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        raise NotImplementedError()

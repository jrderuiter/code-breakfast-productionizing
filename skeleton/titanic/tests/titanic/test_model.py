#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `titanic.model` package."""

import pandas as pd
import pytest

from titanic.model import TitanicModel


class TestTitanicModel:
    """Tests for the TitanicModel class."""

    @pytest.fixture
    def example_dataset(self):
        """Example dataset."""
        X = pd.DataFrame({
            "PClass": [1, 2, 2, 3, 1, 2],
            "Sex": ["male"] * 3 + ["female"] * 3,
        })
        y = pd.Series([0, 0, 1, 1, 1, 1])
        return X, y

    def test_fit(self, example_dataset):
        """Tests fitting the model."""

        X, y = example_dataset

        model = TitanicModel()
        model.fit(X, y)

        assert len(y_pred) > 0

    def test_predict(self, example_dataset):
        """Tests predicting with the model."""

        X, y = example_dataset

        model = TitanicModel()
        model.fit(X, y)

        y_pred = model.predict(X)

        assert len(y_pred) > 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `titanic.features` package."""

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from titanic.features import CabinTransformer


class TestCabinTransformer:
    """Tests for the CabinTransformer."""

    @pytest.fixture
    def example_df(self):
        """Sample pytest fixture."""

        return pd.DataFrame({
            "Cabin": [None, "C37", "E53"]
        })

    def test_example(self, example_df):
        """Tests a simple example."""
        transformer = CabinTransformer()
        result = transformer.transform(example_df)
        assert list(result["Cabin"]) == ["N", "C", "E"]

    def test_does_not_modify_inplace(self, example_df):
        """Tests if the original dataframe is not modified."""

        original_df = example_df.copy()
        result = CabinTransformer().transform(example_df, copy=True)

        assert_frame_equal(example_df, original_df)
        assert list(result["Cabin"]) == ["N", "C", "E"]

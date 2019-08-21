"""Functions for creating the Flask application."""

import io

import pandas as pd
from flask import Flask, Response, request

from titanic.model import Model


class Scorer(Flask):
    """Flask app for scoring predictions using a given model."""

    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_url_rule("/ping", view_func=self.ping)

        # TODO: Load model.
        # TODO: Add /predict route.

    def ping(self):
        """Heartbeat endpoint."""
        return "pong", 200

    def fit(self):
        """Fit endpoint, that re-trains the existing model."""
        # (Bonus) implement a function to re-train the model
        raise NotImplementedError()

    def predict(self):
        """Predict endpoint, which produces predictions for a given dataset."""

        data = pd.read_csv(io.BytesIO(request.get_data()))

        # TODO: Implement model predictions.
        y_pred = [1, 1, 1, 1]
        predictions = pd.DataFrame({"prediction": y_pred})

        return Response(
            predictions.to_csv(None, header=True, index=False), content_type="text/csv"
        )

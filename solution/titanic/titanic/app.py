"""Functions for creating the Flask application."""

import io

import pandas as pd
from flask import Flask, Response, request

from titanic.model import ModelFit


class Scorer(Flask):
    """Flask app for scoring predictions using a given model."""

    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger.info("Loading model from %s", model_path)
        self._model = ModelFit.load(model_path)

        self.add_url_rule("/predict", view_func=self.predict)
        self.add_url_rule("/ping", view_func=self.ping)

    def ping(self):
        """Heartbeat endpoint."""
        return "pong", 200

    def fit(self):
        """Fit endpoint, that re-trains the existing model."""
        # (Bonus) implement a function to re-train the model
        raise NotImplementedError()

    def predict(self):
        """Predict endpoint, which produces predictions for a given dataset."""

        x_predict = pd.read_csv(io.BytesIO(request.data))
        self.logger.info("Loaded %d rows for prediction", x_predict.shape[0])

        y_hat = self._model.predict(x_predict)
        y_hat_df = pd.DataFrame({"predictions": y_hat})

        return Response(
            y_hat_df.to_csv(None, header=True, index=False), content_type="text/csv"
        )

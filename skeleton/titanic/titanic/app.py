"""Functions for creating the Flask application."""

from flask import Flask, request

from titanic.model import ModelFit


class Scorer(Flask):
    """Flask app for scoring predictions using a given model."""

    # TODO: Add logging?

    def __init__(self, *args, model_path,  **kwargs):
        super().__init__(*args, **kwargs)

        self._model = ModelFit.load(model_path)

        # TODO: Add routes using `add_url_rule`.

    def ping(self):
        """Heartbeat endpoint."""
        return "pong", 200

    def fit(self):
        """Fit endpoint, that re-trains the existing model."""
        # (Bonus) implement a function to re-train the model
        raise NotImplementedError()

    def predict(self):
        """Predict endpoint, which produces predictions for a given dataset."""

        payload = request.data
        # assume the payload is the test.csv
        # `PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked`
        #
        # TODO predict the survivors, by including the logic from the model!

        return 'please make me work!'

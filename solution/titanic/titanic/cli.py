# -*- coding: utf-8 -*-

"""Console script for titanic."""

import click

import pandas as pd

from .app import Scorer
from .model import Model, TitanicModel


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.argument("dataset_path")
@click.option("--output_path", default="model.pkl")
@click.option("--label_col", default="Survived")
def fit(dataset_path, output_path, label_col):
    """Fits the model on a given dataset."""

    dataset = pd.read_csv(dataset_path)

    x_train = dataset.drop([label_col], axis=1)
    y_train = dataset[label_col]

    model = TitanicModel()
    model.fit(x_train, y_train)

    model.save(output_path)


@cli.command()
@click.argument("model_path")
@click.argument("dataset_path")
@click.option("--output_path", default="predictions.csv")
def predict(model_path, dataset_path, output_path):
    """Produces predictions for a given dataset."""

    dataset = pd.read_csv(dataset_path)

    model = Model.load(model_path)
    y_hat = model.predict(dataset)

    y_hat_df = pd.DataFrame({"predictions": y_hat})
    y_hat_df.to_csv(output_path, index=False)


@cli.command()
@click.argument("model_path")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5000)
@click.option("--debug/--no-debug", default=False)
def serve(model_path, host, port, debug):
    """Serves a fitted model in a REST API."""

    app = Scorer(__name__, model_path=model_path)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    cli()

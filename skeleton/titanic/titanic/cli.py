# -*- coding: utf-8 -*-

"""Console script for titanic."""

import click


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.argument("dataset_path")
@click.option("--output_path", default="model.pkl")
def fit(dataset_path, output_path):
    """Fits the model on a given dataset."""
    raise NotImplementedError()


@cli.command()
@click.argument("model_path")
@click.option("--port", default=5000)
@click.option("--debug/--no-debug", default=False)
def serve(model_path, port, debug):
    """Serves a fitted model in a REST API."""
    raise NotImplementedError()


if __name__ == "__main__":
    cli()

=======
Titanic
=======

Example solution for the titanic hackathon, in which we aimed to productionize an
ML model for predicting survival on the Titanic dataset.

Installation
------------

The `titanic` package can be installed from the package directory using::

    pip install .

This should also take care of any dependencies required by the library.

Usage
-----

Fitting a model
~~~~~~~~~~~~~~~

The package provides a command line tool `titanic`, which (among others) implements a
`fit` subcommand for fitting a mode. As such, a model can be trained on a given dataset
using the following command::

    titanic fit --output_path model.pkl /path/to/train.csv

In this command, the fitted model is written to a file called `model.pkl`. Note that
currently only one type of model is supported (`TitanicRfModel`). In  future work, we
could work on adding additional models (as additional `Model`  subclasses) and think
about how to support training of different model types from the command line.

Predicting using the CLI
~~~~~~~~~~~~~~~~~~~~~~~~

Once we have a fitted model, we can produce prediction using the `predict` subcommand::

    titanic predict --output_path pred.csv model.pkl ../../data/test.csv

This writes the produced predictions to a csv file called `pred.csv`.

Predicting using the Flask API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fitted model can also be served in a Flask API using the `serve` subcommand::

    titanic serve model.pkl

Note that the `--host` and `--port` options can be used to configure on which
address/port the server is listening.

The Flask API can be used to produce predictions by sending a (CSV) dataset
to the `/predict` endpoint by (for example) sending a request using the Python
library `requests`::

    import io

    import pandas as pd
    import requests

    x_predict_path = "../../data/test.csv"

    with open(x_predict_path, "rb") as file_:
        csv_data = file_.read()

    response = requests.get(
        "http://localhost:5000/predict",
        data=csv_data
    )
    response.raise_for_status()

    y_hat = pd.read_csv(io.BytesIO(response.content))
    print(y_hat)


Running the API in Docker
~~~~~~~~~~~~~~~~~~~~~~~~~

To facilitate running the Flask API in a Docker container we have also provided a
(very simple) example Dockerfile. Using this file, you can build a Docker image for
the API using the following command::

    docker build -t titanic .

This image can be run using `docker run` as follows::

    docker run -v /path/to/model_dir:/model -p 5000:5000 titanic

where `/path/to/model_dir` should point to a directory containing a `model.pkl` file.

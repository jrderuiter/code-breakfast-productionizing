# Hackathon: productionizing predictive models

The goal of this hackathon is to illustrate the process of converting a predictive
machine learning model from a proof-of-concept notebook into a Python package that
can (more easily) be deployed into production. As the hackathon does not focus on
developing the actual model, we will use a 'pre-made' model from Kaggle titanic
challenge https://www.kaggle.com/c/titanic, in which predictive models are trained
to predict the chances of a passengers survival based on their details.

In summary, the overall goals are to:

- Determine the basic building blocks of the titanic model
- Implement these blocks in a Python package, following best practices
- Wrap the model in a REST API using Flask
- Containerize the application using Docker (optional)

## Step 1: run the existing notebook

First, to see what the existing model does, we will start by running the existing
notebook.

- Create a clean environment using conda or virtualenv and install the notebooks
  dependencies into that environment (using the supplied `environment.yml` file
  or the `requirements.txt` file).
- Run the notebook and check it's outputs.
- Can you determine the key steps involved in training the model?
    - Which feature engineering steps are involved?
    - What model is used? With which parameters?

## Step 2: create a basic Python package

In the rest of the hackathon, we will work on developing a Python package. To
experiment, with the basic setup of a Python package, we will first create a
bare bones package using a cookiecutter template.

- Create an empty Python package using the cookiecutter template at:
https://github.com/audreyr/cookiecutter-pypackage.
- Create an empty Python package and install the package (including its dependencies)
  into a clean Python environment.
- Inspect the files that were created by the template.
- Try editing the docs and playing with the different Makefile commands.

## Step 3: create a titanic package using the provided skeleton.

In the `skeleton` folder, we have provided an initial setup for a `titanic` package
that we can use for productionizing the model. We will work on expanding this skeleton
into the fully fledged solution by implementing the model in a scikit-learn pipeline
that can be persisted and loaded to/from disk.

- Expand `features.py` by adding the transformer classes needed for implementing
  the different feature engineering steps of the notebook. Check the scikit-learn
  documentation for existing transformers that you may be able to use before
  implementing your own.
- Add documentation (docstrings) to the added classes. Optional: add basic usage
  examples to `usage.rst` in the Sphinx documentation.
- Implement several unit tests for the added transformers to check if they function as
  expected. Try to use fixtures to share example datasets between tests.
- Implement an additional `Model` class (see `models.py` for more details) that
  implements your model using a scikit-learn pipeline.

## Step 4: wrap the model in a API using Flask

Next, we will wrap our model in a Flask app so that we can serve predictions over
a web API.

- Extend the basic (class-based) Flask app implementation in `app.py` to implement
  a predict endpoint that takes CSV data from a request and returns the corresponding
  predictions.
- To make fitting/running the model easier, implement the `fit` and `serve` commands
  in `cli.py`. The `fit` command should fit and persist a (fitted) model. The `serve`
  command should load a persisted model and serve it using the Flask application.
- Optional: implement a `fit` endpoint that allows re-training of the model using
  a data set passed via a request.

## Step 5: wrap the package in a docker container

Finally, to make running/deploying our package easier, we want to create a docker
image containing our titanic package.

- Create a Dockerfile that installs the package + it's required dependencies and
  serves a model when the container is started.
- Think about how we can serve different model using the same image.

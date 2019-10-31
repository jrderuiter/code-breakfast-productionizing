# Hackathon: productionizing predictive models

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/jrderuiter/code-breakfast-productionizing)

The goal of this hackathon is to illustrate the process of converting a predictive
machine learning model from a proof-of-concept notebook into a Python package that
can (more easily) be deployed into production. As the hackathon does not focus on
developing the actual model, we will use an adapted model from the Kaggle titanic
challenge https://www.kaggle.com/c/titanic, in which predictive models are trained
to predict the chances of a passengers survival based on their details.

In summary, the overall goals of this Hackathon are to:

- Determine the basic building blocks of the titanic model
- Implement these blocks in a Python package (including tests + documentation)
- Wrap the model in a REST API using Flask
- Containerize the application using Docker (optional)

## Step 0: Gettting your machine ready to be able to do the Hackaton

This is an optional step, but we recommend using `conda` for this Hackaton.
There are two options to install `conda`, either install

- Anaconda, https://www.anaconda.com/distribution/#download-section
- Miniconda, https://docs.conda.io/en/latest/miniconda.html

Installing Anaconda is a bit easier, and we would recommend it for first time Python users.
You can verify if conda is installed correctly, by opening a new terminal (or command prompt) and running `conda --version`.
That should print something similar to `conda 4.5.8`.

After installing `conda` you need to either clone or download this repo using the button (green) on the right hand side.

## Step 1: Run the existing notebook

First, to see what the existing model does, we will start by running the existing
notebook.

- Create a clean environment using conda or virtualenv and install the notebooks
  dependencies into that environment (using the supplied `environment.yml` file
  or the `requirements.txt` file).
- Run the notebook and check it's outputs.
- Can you determine the key steps involved in training the model?
    - Which feature engineering steps are involved?
    - What model is used? With which parameters?

Small tip: creating a clean environment with conda is as easy as `conda env create -f environment.yml` from within the notebook folder.
Next, activate your new environment. And run `jupyter notebook`. That should open up a new browser window and allow you to open the notebook (titanic-model.ipynb).


## Step 2: Create a titanic package using the provided skeleton.

In the `skeleton` folder, we have provided an initial setup for a `titanic` package
that we can use for productionizing the model. We will work on expanding this skeleton
into the fully fledged solution by implementing the model in a scikit-learn pipeline
that can be persisted and loaded to/from disk.

- Install the package + its development dependencies into your virtualenv using
  `pip install .[dev]` (run from the titanic package directory) so that you can start
  developing the package.
- An initial setup for the model is provided in `titanic/model.py` by the `TitanicModel`
  class. Expand this class by adding implementations for the fit/predict methods.
- *Optional* - Run pylint over your code using `make lint` to see how well your code
  fits the pylint style. Note that you can modify the rules used by pylint by creating
  a pylintrc configuration file (using `pylint --generate-rcfile > pylintrc`).
- *Optional* - Try running the supplied unit tests using `make tests` to test if your
  implementation passes the supplied (very basic) unit tests.
- *Optional* - Add documentation (docstrings) to the `TitanicModel` class. You can
  generate HTML documentation with Sphinx using the `make docs` command.

## Step 3: Try fitting a model using the provided CLI commands.

We have provided several command-line commands for fitting your model and saving it
to a serialized file, which we can load (and later expose in an API) for producing
predictions.

- Train a model on our train dataset using `titanic fit data/train.csv`. This should
  create a file called `model.pkl` containing your serialized model.
- Try producing predictions using this serialized model using the command
  `titanic predict model.pkl data/test.csv`. This should produce a file called
  `predictions.csv` containing your predictions.

## Step 4: wrap the model in a API using Flask

Next, we will wrap our model in a Flask app so that we can serve predictions over
a web API.

- Try starting the skeleton Flask application using the `titanic serve model.pkl`
  command. You can test the application by opening `http://localhost:5000/ping`
  in your browser. This should display the text `pong` if everything is working
  correctly.
- Extend the basic (class-based) Flask app implementation in `titanic/app.py` to
  implement a `predict` endpoint that takes CSV data from a request and passes it to
  your model. Note that this requires you to load your serialized model and expose
  it in the Scorer class.
- Try sending a dataset to the `predict` endpoint (using a POST request) to see if
  your implementation works. This request should include your prediction dataset as
  data in the request body. (*Tip: useful tools for sending requests are the Postman
  application or the Python library requests. Ask for help if you need more details.*)
- *Optional*: implement a `fit` endpoint that allows re-training of the model using
  a data set passed via a request.

## Step 5: Wrap the package in a docker container

Finally, to make running/deploying our package easier, we want to create a docker
image containing our titanic package.

- Create a Dockerfile that installs the package + it's required dependencies and
  serves a model when the container is started.
- Think about how we can serve different models using the same image.

# -*- coding: utf-8 -*-

"""Classes for creating and persisting (fitted) ML models."""

from sklearn.externals import joblib


class Model:
    """Base class representing a basic model.

    This class defines a generic interface for a machine learning model, which consists
    of a single `fit` method. Implementations of `fit` should take care of fitting
    the model on a specific dataset and return a `ModelFit` instance, which represents
    a read-only model fit that can be used to perform predictions and can be persisted.
    """

    def fit(self, x_train, y_train) -> "ModelFit":
        """Fits model on given dataset.

        Parameters
        ----------
        x_train : pd.Dataframe
            Dataframe containing training data (features only, no response).
        y_train : Union[pd.Series, np.ndarray]
            Pandas series (or numpy array) containing the response values for the
            given training dataset.

        Returns
        -------
        ModelFit
            A trained model instance.

        """
        raise NotImplementedError()


class ModelFit:
    """Basic class representing a trained model (a model fit).

    Model fits are in principle read-only and should not be modified after training.
    As such, instances represent trained 'snapshots' of the model that can be persisted
    to disk and loaded as needed for performing predictions. Optionally, model fits
    can store extra metadata/parameters to indicate how/when the model was trained.
    """

    def predict(self, x_predict):
        """Returns predictions for a given dataset.

        Parameters
        ----------
        x_predict : pd.DataFrame
            Dataframe to perform predictions for. Should follow the same structure
            as the dataset on which the original model was trained.

        Returns
        -------
        np.ndarray
            Array containing predictions.

        """
        raise NotImplementedError()

    @classmethod
    def load(cls, file_path):
        """Loads model fit from given file path.

        Parameters
        ----------
        file_path : str
            Path to a pickled model file.

        Returns
        -------
        ModelFit
            The unpickled model instance.

        """
        return joblib.load(file_path)

    def save(self, file_path):
        """Saves model fit to given file path.

        Parameters
        ----------
        file_path : str
            Path to save the pickled model to.

        """
        joblib.dump(self, file_path)


class SklearnModelFit(ModelFit):
    """ModelFit class for scikit-learn models.

    ModelFit class representing a trained scikit-learn model. Can be used to wrap
    any type of model that follows the scikit-learn fit/predict interface.

    Parameters
    ----------
    model
        A trained scikit-learn model.

    """

    def __init__(self, model):
        self._model = model

    def predict(self, x_predict):
        return self._model.predict(x_predict)

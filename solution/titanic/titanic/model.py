# -*- coding: utf-8 -*-

"""Classes for creating and persisting (fitted) ML models."""

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

from . import features


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

    def predict(self, x_predict) -> np.ndarray:
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


class TitanicRfModel(Model):
    """A RandomForest-based model for predicting survival in the Titanic dataset."""

    # TODO: Make model more configurable with parameters in the constructor.

    def fit(self, x_train, y_train):
        # Build classifier pipeline.
        pipeline = self._build_pipeline()

        # Choose some parameter combinations to try
        parameters = {
            "rf__n_estimators": [4, 6, 9],
            # "rf__max_features": ["log2", "sqrt", "auto"],
            # "rf__criterion": ["entropy", "gini"],
            # "rf__max_depth": [2, 3, 5, 10],
            # "rf__min_samples_split": [2, 3, 5],
            # "rf__min_samples_leaf": [1, 5, 8],
        }

        # Type of scoring used to compare parameter combinations
        acc_scorer = metrics.make_scorer(metrics.accuracy_score)

        # Run the grid search
        grid_obj = GridSearchCV(pipeline, parameters, scoring=acc_scorer, cv=5)
        grid_obj = grid_obj.fit(x_train, y_train)

        # Set the clf to the best combination of parameters
        clf = grid_obj.best_estimator_

        # Fit the best algorithm to the data.
        clf.fit(x_train, y_train)

        return SklearnModelFit(model=clf)

    def _build_pipeline(self):
        """Helper method for building the model pipeline."""

        return Pipeline(
            steps=[
                # Bin fares/ages into numeric categories.
                (
                    "bin_fares_ages",
                    features.BinTransformer(
                        bins={
                            "Fare": [-1, 0, 8, 15, 31, 1000],
                            "Age": [-1, 0, 5, 12, 18, 25, 35, 60, 120],
                        },
                        fill_values={"Fare": -0.5, "Age": -0.5},
                    ),
                ),
                # Simplify cabin features into categories.
                ("simplify_cabins", features.CabinTransformer()),
                # Encode sex/cabin into numeric columns.
                (
                    "ordinal_encoder",
                    features.OrdinalEncoder(
                        categories={
                            "Sex": ["male", "female"],
                            "Cabin": ["N", "C", "E", "G", "D", "A", "B", "F", "T"],
                        }
                    ),
                ),
                # Final column selection.
                (
                    "select_features",
                    features.SelectColumnTransformer(
                        columns=[
                            "Pclass",
                            "Sex",
                            "Age",
                            "SibSp",
                            "Parch",
                            "Fare",
                            "Cabin",
                        ]
                    ),
                ),
                # Use a RandomForest classifier.
                ("rf", RandomForestClassifier()),
            ]
        )

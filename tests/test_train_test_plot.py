from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import linear_model
from numpy import random



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


def train_test_plot():
    """
    Creates plot of training and test error for an arbitrary sklearn model.

    Args:
        model (sklearn object): Previously initialized, untrained sklearn classifier or regression object
                                    with fit & predict methods. Can also be a pipeline with several steps.

        score_type (list or str): Should be one of: [mse, r2, adj_r2, auc, ...].
                                      If a list, then a list containing several of those entries as elements.

        x (ndarray): (n x d) array of features.
        y (ndarray): (n x 1) Array of labels

        hyperparameter (string): Hyperparameter of model to iterate over creating plot

        param_range (list): Range of hyperparameter values to iterate over

        random_seed (int): Default = None. If set to integer, defines the random train_test_split

        verbose (boolean): Default = `False`. If set to `True` returns list of training and test score.
                           Added for plot testing.


    Returns:
        none. Calls plt.show() to display plot


    """
    pass


class TestClass:

    def test_train_test_plot_tree(self):

        train_score_list = []
        val_score_list = []

        X = digits['data']
        y = digits['target']

        for i in range(1, 20):
            model = DecisionTreeClassifier(max_depth=i)
            scores = Score(model, 'accuracy', x=X, y=y, random_seed=1234)
            train_score = scores.scores[0]
            val_score = scores.scores[1]

            train_score_list.append(train_score)
            val_score_list.append(val_score)

        exp = (train_score_list, val_score_list)

        obs = train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y,
                              hyperparameter="max_depth",
                              param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        assert obs == exp


    def test_train_test_plot_ridge(self):

        random_seed = 1234
        n_samples, n_features = 10, 5
        y = np.random.randn(n_samples)
        X = np.random.randn(n_samples, n_features)

        train_score_list = []
        val_score_list = []

        for i in np.linspace(2 ** -2, 2 ** 2, 10):
            model = Ridge(alpha=i)
            scores = Score(model, 'mse', x=X, y=y, random_seed=1234)
            train_score = scores.scores[0]
            val_score = scores.scores[1]

            train_score_list.append(train_score)
            val_score_list.append(val_score)

            exp = (train_score_list, val_score_list)

            obs = train_test_plot(model=Ridge(), score_type="mse", x=X, y=y, hyperparameter="alpha",
                                  param_range=list(np.linspace(2 ** -2, 2 ** 2, 10)), random_seed=1234, verbose=True)

            assert exp == obs

    def train_test_lasso_r2(self):
        random_seed = 1234
        n_samples, n_features = 10, 5
        y = np.random.randn(n_samples)
        X = np.random.randn(n_samples, n_features)

        for i in np.linspace(2 ** -2, 2 ** 2, 10):
            model = Lasso(alpha=i)
            scores = Score(model, 'r2', x=X, y=y, random_seed=1234)
            train_score = scores.scores[0]
            val_score = scores.scores[1]

            train_score_list.append(train_score)
            val_score_list.append(val_score)

        exp = (train_score_list, val_score_list)

        obs = train_test_plot(model=Lasso(), score_type="r2", x=X, y=y,
                              hyperparameter="alpha",
                              param_range=list(list(np.linspace(2 ** -2, 2 ** 2, 10))), random_seed=1234, verbose=True)

        def train_test_lasso_adjr2(self):
            random_seed = 1234
            n_samples, n_features = 10, 5
            y = np.random.randn(n_samples)
            X = np.random.randn(n_samples, n_features)

            for i in np.linspace(2 ** -2, 2 ** 2, 10):
                model = Lasso(alpha=i)
                scores = Score(model, 'adj_r2', x=X, y=y, random_seed=1234)
                train_score = scores.scores[0]
                val_score = scores.scores[1]

                train_score_list.append(train_score)
                val_score_list.append(val_score)

            exp = (train_score_list, val_score_list)

            obs = train_test_plot(model=Lasso(), score_type="r2", x=X, y=y,
                                  hyperparameter="alpha",
                                  param_range=list(list(np.linspace(2 ** -2, 2 ** 2, 10))), random_seed=1234,
                                  verbose=True)

    def test_no_input(self):
        """ Ensures that TypeError is raised if either of the inputs is missing/wrong. """

        import pytest

        # Test A
        with pytest.raises(TypeError):
            train_test_plot(model=None, score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        # Test B
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=None, y=y,
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        # Test C
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=None,
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        # Test D
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type=None, x=X, y=y, hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        # Test E
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter=None,
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        # Test F
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=None, random_seed=1234, verbose=True)
        # Test G
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=None, random_seed=None, verbose=True)

        # Test H
        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=None, random_seed=1234, verbose=None)


    def test_input_shape(self):
        with pytest.raises(RuntimeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=np.array([1, 1, 1, 1]),
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)


    def test_wrong_parameter(self):
        """ Ensures that ValueError if hyperparamter does not match model"""

        with pytest.raises(ValueError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter="alpha",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(ValueError):
            train_test_plot(model=Ridge(), score_type="mse", x=X, y=y, hyperparameter="max_depth",
                            param_range=list(np.linspace(2 ** -2, 2 ** 2, 10)), random_seed=1234, verbose=True)


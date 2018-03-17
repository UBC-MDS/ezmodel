from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import linear_model
from numpy import random


import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestClass:

    def test_train_test_plot_tree(self):

        train_score_list = []
        val_score_list = []

        digits = load_digits()
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

        assert (abs((np.array(obs) - np.array(exp))) < 0.05).all()

    #Test J
    def test_wrong_hyp(self):
        with pytest.raises(ValueError):
            train_test_plot(model=RFC(), score_type="accuracy", x=X, y=y,
                            hyperparameter="cp",
                            param_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], random_seed=1234)


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

    def test_train_test_lasso_r2(self):

        train_score_list = []
        val_score_list = []

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

        assert (np.array(exp) == np.array(obs)).all()

        def train_test_lasso_adjr2(self):

            train_score_list = []
            val_score_list = []

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

            obs = train_test_plot(model=Lasso(), score_type="adj_r2", x=X, y=y,
                                  hyperparameter="alpha",
                                  param_range=list(list(np.linspace(2 ** -2, 2 ** 2, 10))), random_seed=1234,
                                  verbose=True)

            assert (np.array(exp) == np.array(obs)).all()


    def test_no_input(self):
        """ Ensures that TypeError is raised if either of the inputs is missing/wrong. """

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
        """Tests if data input is of the right shape"""
        with pytest.raises(RuntimeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=np.array([1, 1, 1, 1]),
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)


    def test_wrong_combination(self):
        """ Ensures that ValueError if hyperparamter does not match model"""

        # Test K
        with pytest.raises(ValueError):
            train_test_plot(model=Ridge(), score_type="mse", x=X, y=y, hyperparameter="max_depth",
                            param_range=list(np.linspace(2 ** -2, 2 ** 2, 10)), random_seed=1234, verbose=True)

        # Test L
        with pytest.raises(ValueError):
            train_test_plot(model=Lasso(), score_type="mse", x=X, y=y,
                            hyperparameter="cp",
                            param_range=list(list(np.linspace(2 ** -2, 2 ** 2, 10))), random_seed=1234,
                            verbose=True)
        #Test M
        with pytest.raises(ValueError):
            train_test_plot(model=Lasso(), score_type="mse", x=X, y=y,
                            hyperparameter="cp",
                            param_range=list(list(np.linspace(2 ** -2, 2 ** 2, 10))), random_seed=1234,
                            verbose=True)


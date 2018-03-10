from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        random_seed = 1234
        digits = datasets.load_digits()

        X = digits['data']  # these are the features (pixel intensities). split this into X_train and X_validation
        y = digits['target']  # these are the labels (0-9). split this into y_train and y_validation.

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        train_score_list = []
        val_score_list = []

        # iterating over range of different values for hyperparameter while keeping track of score
        for i in np.linspace(2 ** -2, 2 ** 2, 10):
            tree = DecisionTreeClassifier(max_depth=i)
            tree.fit(X_train, y_train)
            train_score = tree.score(X_train, y_train)
            val_score = tree.score(X_validation, y_validation)
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

        mse_train_list = []
        mse_test_list = []

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=random_seed)

        # iterating over range of different values for hyperparameter while keeping track of score
        for i in np.linspace(2 ** -2, 2 ** 2, 10):
            model = Ridge(alpha=i, fit_intercept=True, normalize=False, copy_X=True,
                          max_iter=None, tol=0.001, solver='auto', random_state=None)
            model.fit(X_train, y_train)

            mean_squared_err = lambda y, yhat: np.mean((y - yhat) ** 2)
            errors = [mean_squared_err(y_train, model.predict(X_train)),
                      mean_squared_err(y_validation, model.predict(X_validation))]

            mse_train_list.append(errors[0])
            mse_test_list.append(errors[1])

            exp = (mse_train_list, mse_test_list)

            obs = train_test_plot(model=Ridge(), score_type="mse", x=X, y=y, hyperparameter="alpha",
                                  param_range=list(np.linspace(2 ** -2, 2 ** 2, 10)), random_seed=1234, verbose=True)

            assert exp == obs


    def test_no_input(self):
        """ Ensures that TypeError is raised if either of the inputs is missing/wrong. """

        with pytest.raises(TypeError):
            train_test_plot(model=None, score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type=None, x=X, y=y, hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=None, y=y,
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=None,
                            hyperparameter="max_depth",
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter=None,
                            param_range=list(range(1, 20)), random_seed=1234, verbose=True)

        with pytest.raises(TypeError):
            train_test_plot(model=DecisionTreeClassifier(), score_type="accuracy", x=X, y=y, hyperparameter="max_depth",
                            param_range=None, random_seed=None, verbose=True)
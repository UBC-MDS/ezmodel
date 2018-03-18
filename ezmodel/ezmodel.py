#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn import clone
from ezmodel.Score import Score

def train_test_plot(model, score_type,
                    x, y, hyperparameter, param_range, random_seed, verbose=False):
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

    random_seed = random_seed

    # testing input type for parameters

    # Condition A
    if str(type(model))[8:15] != 'sklearn':
        raise TypeError("model must be an sklearn model")

    # Condition B
    if str(type(x))[14:21] != 'ndarray':
        raise TypeError("x must be a numpy array")

    # Condition C
    if str(type(y))[14:21] != 'ndarray':
        raise TypeError("y must be a numpy array")

    # Condition D
    if x.shape[0] != y.shape[0]:
        raise RuntimeError("the number of rows in x and y has to be equal")

    # Condition E
    if not isinstance(score_type, str):
        raise TypeError("score_type must be a string")

    # Condition F
    if not isinstance(hyperparameter, str):
        raise TypeError("hyperparameter must be a string")

    # Condition G
    if not isinstance(param_range, list):
        raise TypeError("param_range must be a list")

    # Conition H
    if not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")

    # Condition I
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be an boolean")

    # Condition J
    if score_type not in ["r2", "mse", "r2", "adj_r2", "accuracy"]:
        raise ValueError("score_type must be one of: 'r2', 'mse', 'r2', 'adj_r2', 'accuracy'")

    # initiating lists for plot
    train_score_list = []
    val_score_list = []
    index_list = []

    # Condition K
    if score_type == "accuracy":

        # iterating over range of different values for hyperparameter while keeping track of score
        try:
            for i in param_range:
                model = model.set_params(**{str(hyperparameter): i})
                scores = Score(model, 'accuracy', x=x, y=y, random_seed=random_seed)
                train_score = scores.scores[0]
                val_score = scores.scores[1]

                train_score_list.append(train_score)
                val_score_list.append(val_score)

                index_list.append(i)

        except ValueError:
            raise ValueError("Please, check sklearn docs for correct combination of 'model' and 'hyperparameter'")
            return

            # Condition L
    if score_type == "mse":

        try:
            for i in param_range:
                model = model.set_params(**{str(hyperparameter): i})
                scores = Score(model, 'mse', x=x, y=y, random_seed=random_seed)
                train_score = scores.scores[0]
                val_score = scores.scores[1]

                train_score_list.append(train_score)
                val_score_list.append(val_score)

                index_list.append(i)

        except ValueError:
            raise ValueError("Please, check sklearn docs for correct combination of 'model' and 'hyperparameter'")
            return

            # Condition M
    if score_type == "r2":

        try:
            for i in param_range:
                model = model.set_params(**{str(hyperparameter): i})
                scores = Score(model, 'r2', x=x, y=y, random_seed=random_seed)
                train_score = scores.scores[0]
                val_score = scores.scores[1]

                train_score_list.append(train_score)
                val_score_list.append(val_score)

                index_list.append(i)

        except ValueError:
            raise ValueError("Please, check sklearn docs for correct combination of 'model' and 'hyperparameter'")
            return

            # Condition N
    if score_type == "adj_r2":

        try:
            for i in param_range:
                model = model.set_params(**{str(hyperparameter): i})
                scores = Score(model, 'adj_r2', x=x, y=y, random_seed=random_seed)
                train_score = scores.scores[0]
                val_score = scores.scores[1]

                train_score_list.append(train_score)
                val_score_list.append(val_score)

                index_list.append(i)

        except ValueError:
            raise ValueError("Please, check sklearn docs for correct combination of 'model' and 'hyperparameter'")
            return

            # This if statement is only included for testing purposes - no test will be written
    if verbose == True:
        return train_score_list, val_score_list


    else:
        plt.plot(index_list, train_score_list, label="training {}".format(score_type))
        plt.plot(index_list, val_score_list, label="test {}".format(score_type))
        plt.legend()

        return



def regularization_plot(model, alpha, x, y, tol=1e-7):
    """
     Plots coeffiecients from results of Lasso, Ridge, or Logistic Regression model


    Args:
        model (sklearn object): Previously initialized, untrained sklearn regression object
                                with fit & predict methods. Model has to be one of the following:
                                LogisticRegression(), Ridge(), Lasso().
                                Can also be a pipeline with several steps containing one of the above models.

        alpha: Penalty constant multiplying the regularization term. Larger value corresponds to stronger
                regularization. Can be list or float/int.

        x (ndarray): (n x d) array of features.

        y (ndarray): (n x 1) Array of labels.

        tol (float): coefficients less than this will be treated as zero.

    Returns:
        Calls plt.show() to display plot. Plot shown depends on type of alpha argument: If list, displays and returns number
        of non-zero features for each alpha value; if float/int: returns and displays coefficient magnitudes.

    """
    # conditional A
    if type(model) not in [type(Lasso()), type(Ridge()), type(LogisticRegression())]:
        raise TypeError("Model specified must have same type as Lasso(), Ridge(), or LogisticRegression()")

    # conditional B
    if not isinstance(alpha, list):
        # conditional C
        if isinstance(model, type(LogisticRegression())):
            fitted_model = model.set_params(**{'C':1/alpha})
            fitted_model.fit(x,y)
            coefs = fitted_model.coef_[0]
        else:
            fitted_model = model.set_params(**{'alpha':alpha})
            fitted_model.fit(x,y)
            coefs = fitted_model.coef_
        coefs = [np.abs(c) if c>tol else 0 for c in coefs]

        plt.plot(range(len(coefs)), coefs, linestyle='-', marker='o')
        plt.title("Magnitude of Nonzero Coefficients")

    else:
        # Conditional D
        if isinstance(model, type(LogisticRegression())):
            fitted_models = [clone(model).set_params(**{'C':1/a}) for a in alpha]
            fitted_models = [m.fit(x,y) for m in fitted_models]
            coefs = [sum(np.abs(m.coef_[0]) > tol) for m in fitted_models]

        else:
            fitted_models = [clone(model).set_params(**{'alpha':a}) for a in alpha]
            fitted_models = [m.fit(x,y) for m in fitted_models]
            coefs = [sum(np.abs(m.coef_) > tol) for m in fitted_models]

        plt.plot(alpha, coefs, linestyle='-', marker='o')
        plt.title("Number of Nonzero Coefficients")

    return coefs



def _coerce(x):
    """
    Utility function to coerce data into the correct types to be passed to sklearn

    Args:
        x (??): Data to be passed to a model.

    Returns:
        np.ndarray containing the data from x

    Notes:
        Works for pandas DataFrames and nested lists currently. Investigating other types that need coercing.
    """
    if isinstance(x, pd.DataFrame):
        return pd.as_matrix(x)
    elif isinstance(x, list):
        return np.asarray(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError("{} is currently not supported. Please input a numpy array, pandas DataFrame, or nested list".format(type(x)))
    else:
        return x

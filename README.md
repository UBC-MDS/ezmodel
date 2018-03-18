# ezmodel/ezmodelR
[![Build Status](https://travis-ci.org/UBC-MDS/ezmodel.svg?branch=master)](https://travis-ci.org/UBC-MDS/ezmodel)

# Contributors
* Alexander Kleefeldt
* Sean Conley
* Tyler Roberts

# Summary

This package extends the functionality of `sklearn`/`caret`, specifically with regards to model diagnostics,
adding automation and flexibility.

One goal of this package is to provide a functional API to automate common machine learning workflows such as splitting data,
fitting and evaluating models, as well as visualizing the results to allow for easier interpretation and hyperparameter tuning.

Further, this package will address specific omissions of the `sklearn` and `caret` packages and "patch up" these omissions
by adding functionality. One example for this is the limited scope of the `score` function
in `sklearn` which only returns the unadjusted r-squared value. Here, the package will allow the
user to choose different scoring functions based on the problem at hand.


# List of Functions

1. `train_test_plot()`: Simply and easily visualize the training and validation error of your model, allowing you to spend more time protoyping and less time writing boilerplate code.

2. `Score()`: Scoring class to allow a user more control over the scores used to validate their model's performance. Currently includes: accuracy, mse, specificity, sensitivity, $R^2$, and adjusted $R^2$.

3. `regularization_plot()`  Visualize the results of L1 or L2 regularization on your linear models to help determine the optimal strength of your regularization parameter.

# Usage

Usage of the functions is straightforward, and examples are below.

```{python}
import ezmodel

# compare training and test error for sklearn model vs. a hyperparameter
# verbose = True so errors are returned
train_test_plot(Lasso(), 'mse', x, y, 'alpha', [2,4,8], random_seed=123)

# plot coefficient magnitude with regularization strength 2
# coefficients less than tol will be treated as 0
regularization_plot(Ridge(), alpha=2, x, y, tol=1e-6)

# calculate adjusted r2 score for model
Score(LinearRegression(), 'adj_r2', x, y)
```

# Installation

Installation is straightforward with `pip`:

```
pip install git+git://github.com/UBC-MDS/ezmodel
```

# Description of Landscape

There exists a limited ability to complete all of these tasks within both `sklearn` and `caret`, but they require user-defined functions that utilize manually
extracted data (e.g. coefficients, predictions, etc.), or only offer limited diagnostics (e.g. unadjusted R^2 scores). Users of these packages frequently find
themselves repeating the same workflow, for example, splitting the dataset, training the model, and plotting training/validation error. This package will
streamline this process.

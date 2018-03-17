import pytest
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.datasets import load_boston, load_breast_cancer
from ezmodel import ezmodel, Score


def test_input_model_type():
    """Checks TypeError is raised when input model isn't of allowed type."""
    X, y = load_boston(return_X_y=True)
    with pytest.raises(TypeError):
        regularization_plot(LinearRegression(), alpha=1, x=X, y=y)


def test_nonzero_count_ridge():
    """Checks using list for alpha with Ridge() outputs correct coefficient counts."""
    X, y = load_boston(return_X_y=True)
    tol=1e-2
    alpha_range = [2**i for i in range(-2, 3)]
    ridge_models = [Ridge(alpha=a).fit(X,y) for a in alpha_range]
    nz_count = [sum(np.abs(m.coef_)>tol) for m in ridge_models]

    assert nz_count == regularization_plot(Ridge(), alpha=alpha_range, tol=tol, x=X, y=y)


def test_nonzero_count_lasso():
    """Checks using list for alpha with Lasso() outputs correct coefficient counts."""
    X, y = load_boston(return_X_y=True)
    tol=1e-6
    alpha_range = [2**i for i in range(-2, 3)]
    lasso_models = [Lasso(alpha=a).fit(X,y) for a in alpha_range]
    nz_count = [sum(np.abs(m.coef_)>tol) for m in lasso_models]

    assert nz_count == regularization_plot(Lasso(), alpha=alpha_range, tol=tol, x=X, y=y)


def test_nonzero_count_logistic():
    """Checks using list for alpha with LogisticRegression() outputs correct coefficient counts."""
    X, y = load_breast_cancer(return_X_y=True)
    tol=1e-5
    C_range = [2**i for i in range(-2, 3)]
    log_models = [LogisticRegression(C=k).fit(X,y) for k in C_range]
    nz_count = [sum(np.abs(m.coef_[0])>tol) for m in log_models]

    assert nz_count == regularization_plot(LogisticRegression(), alpha=C_range, tol=tol, x=X, y=y)

def test_nonzero_coefs_logistic():
    """Checks using int for alpha produces correct coefficients for LogisticRegression() model."""
    X, y = load_breast_cancer(return_X_y=True)
    tol=1e-7
    mod = LogisticRegression(C=10.0**-7).fit(X,y)
    mod_coefs = mod.coef_[0]
    mod_coefs = [np.abs(c) if c>tol else 0 for c in mod_coefs]

    assert mod_coefs == regularization_plot(LogisticRegression(), alpha=10.0**7, x=X, y=y)

def test_nonzero_coefs_rigde():
    """Checks using float for alpha produces correct coefficients for Ridge() model."""
    X, y = load_boston(return_X_y=True)
    tol=1e-6
    mod=Ridge(alpha=2**2.0).fit(X,y)
    mod_coefs = mod.coef_
    mod_coefs = [np.abs(c) if c>tol else 0 for c in mod_coefs]

    assert mod_coefs == regularization_plot(Ridge(), alpha=2**2.0, x=X, y=y)

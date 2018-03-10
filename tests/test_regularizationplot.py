import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.datasets import load_boston, load_breast_cancer


def test_input_model_type():
    """Checks TypeError is raised when input model isn't of allowed type."""
        X, y = load_boston(return_X_y=True)
        with pytest.raises(TypeError):
            regularization_plot(LinearRegression(), alpha=1, X, y)


def test_nonzero_count_ridge():
    """Checks using list for alpha with Ridge() outputs correct coefficient counts."""
    X, y = load_boston(return_X_y=True)
    tol=1e-2
    alpha_range = [2**i for i in range(-2, 3)]
    ridge_models = [Ridge(alpha=a).fit(X,y) for a in alpha_range]
    nz_count = [sum(np.abs(m.coef_)>tol) for m in ridge_models]

    assert nz_count == regularization_plot(Ridge(), alpha=alpha_range, tol=tol, X, y)


def test_nonzero_count_lasso():
    """Checks using list for alpha with Lasso() outputs correct coefficient counts."""
    X, y = load_boston(return_X_y=True)
    tol=1e-6
    alpha_range = [2**i for i in range(-2, 3)]
    lasso_models = [Lasso(alpha=a).fit(X,y) for a in alpha_range]
    nz_count = [sum(np.abs(m.coef_)>tol) for m in lasso_models]

    assert nz_count == regularization_plot(Lasso(), alpha=alpha_range, tol=tol, X, y)


def test_nonzero_coefs_logistic():
    """Checks using float for alpha produces correct coefficients for LogisticRegression() model."""
    X, y = load_breast_cancer(return_X_y=True)
    tol=1e-7
    mod = LogisticRegression(C=10**-7).fit(X,y)
    mod_coefs = np.abs(mod.coef_[0])
    for i in range(len(mod_coefs)):
        if mod_coefs[i] < tol:
            mod_coefs[i]=0

    assert mod_coefs == regularization_plot(LogisticRegression(), alpha=10**7, X, y)

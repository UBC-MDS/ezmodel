
### `regularization_plot`

Conditional statements and corresponding test from top to bottom:

|`if` statement|Corresponding test|
|--------------|------------------|
|`if type(model) not in [type(Lasso()), type(Ridge()), type(LogisticRegression())]:`| `test_input_model_type()` |
|`if not isinstance(alpha, list):`|`test_nonzero_coefs_logistic():`|
|`if isinstance(model, type(LogisticRegression())):`|`test_nonzero_coefs_logistic():`|
|`else:`|`test_nonzero_coefs_ridge()`|
|`else:`|`test_nonzero_count_logistic():`|
|`if isinstance(model, type(LogisticRegression())):`|`test_nonzero_count_logistic():`|
|`else:`|`test_nonzero_count_ridge():`|

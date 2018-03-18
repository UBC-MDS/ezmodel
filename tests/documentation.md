
### `regularization_plot`

Conditional statements and corresponding test, as labelled in the code. Labelled A-D, where `else` statement often corresponds to `False` outcome.

|Statement| True Test|False Test|
|---------|-------------------|
|A|`test_input_model_type`|`test_nonzero_count_ridge`|
|B|`test_nonzero_coefs_logistic`|`test_nonzero_count_logistic`|
|C|`test_nonzero_coefs_logistic`|`test_nonzero_coefs_rigde`|
|D|`test_nonzero_count_logistic`|`test_nonzero_count_lasso`|

### `train_test_plot`

Conditional statements and corresponding test, as labelled in the code. Branches are labelled A-M.


|Statement|True Test|False Test|
|---|---|---|
|A|`test_no_input` A|`test_train_test_plot_tree`|
|B|`test_no_input` B|`test_train_test_plot_tree`|
|C|`test_no_input` C|`test_train_test_plot_tree`|
|D|`test_input_shape`|`test_train_test_plot_tree`|
|E|`test_no_input` D|`test_train_test_plot_tree`|
|F|`test_no_input` E|`test_train_test_plot_tree`|
|G|`test_no_input` F|`test_train_test_plot_tree`|
|H|`test_no_input` G|`test_train_test_plot_tree`|
|I|`test_no_input` H|`test_train_test_plot_tree`|
|J|`test_no_input` D|`test_train_test_plot_tree`|
|K|`test_train_test_plot_tree`|`test_train_test_plot_lasso_r2`|
|L|`test_train_test_plot_ridge`|`test_train_test_plot_lasso_r2`|
|M|`test_train_test_plot_lasso_r2`|`test_train_test_plot_lasso_adjr2`|
|N|`test_train_test_plot_lasso_adjr2`|`test_train_test_plot_tree`|



### Integration Test

Please, note that `regularization_plot` does not interact with any other functions in the package.  

`train_test_plot` uses the score computed by the `Score` function of the package. Thereofore, the following functions test for smooth interaction of `Score` and `test_train_plot`:

|Score Type|Test|
|---|---|
|accuracy|`test_train_test_plot_tree`|
|r2|`test_train_test_plot_lasso_r2`|
|adjusted r2|`test_train_test_plot_lasso_adjr2`|
|mse|`test_train_test_plot_ridge`|

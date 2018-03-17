
### `regularization_plot`

Conditional statements and corresponding test, as labelled in the code. Labelled A-D, where `else` statement often corresponds to `False` outcome.

|Statement| True Test|False Test|
|---------|-------------------|
|A|`test_input_model_type`|`test_nonzero_count_ridge`|
|B|`test_nonzero_coefs_logistic`|`test_nonzero_count_logistic`|
|C|`test_nonzero_coefs_logistic`|`test_nonzero_coefs_rigde`|
|D|`test_nonzero_count_logistic`|`test_nonzero_count_lasso`|

### `train_test_split`

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

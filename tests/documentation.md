
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


|Statement|Test|
|---|---|
|A|`test_no_input` A|
|B|`test_no_input` B|
|C|`test_no_input` C|
|D|`test_input_shape`|
|E|`test_no_input` D|
|F|`test_no_input` E|
|G|`test_no_input` F|
|H|`test_no_input`G|
|I|`test_no_input`H|
|J|`test_train_test_plot_tree`|
|K|`test_train_test_plot_ridge`|
|L|`test_train_test_plot_lasso_r2`|
|M|`test_train_test_plot_lasso_adjr2`|

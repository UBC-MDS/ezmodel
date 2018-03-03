def test_train_plot(model, score_type, x, y, hyperparameter, param_range):
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


    Returns:
        None. Calls plt.show() to display plot


    """
    pass


def regularization_plot(model, alpha, tol=1e-7, x, y):
    """
     Plots coeffiecients from results of Lasso, Ridge, or Logistic Regression model


    Args:
        model (sklearn object): Previously initialized, untrained sklearn regression object
                                with fit & predict methods. Model has to be one of the following:
                                LogisticRegression(), Ridge(), Lasso().
                                Can also be a pipeline with several steps containing one of the above models.

        alpha (float or list): Penalty constant multiplying the regularization term. Larger value corresponds to stronger
                  regularization. Can be list or float.

        tol (float): Coefficients less than this will be treated as zero if a list is is given for alpha argument.

        x (ndarray): (n x d) array of features.
        y (ndarray): (n x 1) Array of labels

    Returns:
        List of counts of non-zero coefficients if alpha is a list, or list of coefficients if alpha is a float.
        Calls plt.show() to display plot.
        Plot shown depends on type of alpha argument. If list, then returns plot of non-zero features,
        and if float, then displays plot of coefficients' magnitudes.

    """
    pass


class Score(object):
    """ Scoring object. Allows computation of an arbitrary score metric on an arbitrary sklearn model. """

    def __init__(self, model, score_type='mse', x=None, y=None, random_seed=None):
        """
        Constructor for score object. Adds in the model as well as the score type you are looking for.
        # Could score_type be a list?

        Optionally, you could add in x and y values here, to immediately compute a score.


        Args:
            model (sklearn object): Previously initialized, untrained sklearn classifier or regression object
                                    with fit & predict methods. Can also be a pipeline with several steps.

            score_type (list or str): Default='mse'. Should be one of: [mse, r2, adj_r2, auc, ...].
                                      If a list, then a list containing several of those entries as elements.
            x (ndarray): Default=None. (n x d) array of features.
            y (ndarray): Default=None. (n x 1) Array of labels
            random_state (int): Random state for train_test split and model (if required). Passed to sklearn functions.

        Returns:
            None. Sets attributes of the score function, and if the optional values are provided, computes the score.
        """
        pass

    # def __str__(self):
    #     """ Overwrite __str__ method to print information about the scores contained in the object when called."""
    #     pass

    def _accuracy(self):
        """ Computes Accuracy of a model. Number of correct predictions over total number of predictions.
            Uses self.model, self.x and self.y """
        pass

    def _mse(self):
        """ Computes Mean Squared Error. Uses self.model, self.x and self.y"""
        pass

    def _r2(self):
        """ Computes R-Squared. Uses self.model, self.x and self.y """
        pass

    def _adj_r2(self):
        """ Computes Adjusted R-Squared. Uses self.model, self.x and self.y """
        pass

    def _auc(self):
        """ Computes Area Under the Receiver Operator Curve. Uses self.model, self.x and self.y"""
        pass

    def _sensitivity(self):
        """ Computes model sensitivity. Used for AUC. Uses self.model, self.x and self.y """
        pass

    def _specificity(self):
        """ Computes model specificity. Used for AUC. Uses self.model, self.x and self.y """
        pass

    def calculate(self, x, y, score_type=None):
        """
        Computes values for scores if x and y were not provided at intialization.

        Args:
            x (ndarray): (n x d) array of features.
            y (ndarray): (n x 1) array of labels.
            score_type (list or str): Default=self.score_type. Should be one of: [mse, r2, adj_r2, auc, ...].
                                      If a list, then a list containing several of those entries as elements.

        Returns:
            scores (list or dict): If score_type is a list, then keys are score_type, values are a list containing
                                   training and validation errors.
                                   If score_type is a string, then scores is a list containing two elements:
                                   training and validation error.
        """
        # if score_type is None:
        #   score_type = self.score_type
        pass

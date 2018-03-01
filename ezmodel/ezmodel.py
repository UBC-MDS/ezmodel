def test_train_plot():
    """"""
    pass


def feature_importance_plot():
    """"""
    pass


class Score(object):
    """ Scoring object. Allows computation of an arbitrary score metric on an arbitrary sklearn model. """

    def __init__(self, model, score_type='mse', x=None, y=None):
        """
        Constructor for score object. Adds in the model as well as the score type you are looking for.
        # Could score_type be a list?

        Optionally, you could add in x and y values here, to immediately compute a score.


        Args:
            model (sklearn object): Previously initialized sklearn classifier or regression object with fit & predict methods.
                                    Can also be a pipeline with several steps.

            score_type (list or str): Should be one of: [mse, r2, adj_r2, auc, ...].
                                      If a list, then a list containing several of those entries as elements.
            x (ndarray): (d x n) array of features.
            y (ndarray): (1 x n) Array of labels

        Returns:
            None. Sets attributes of the score function, and if the optional values are provided, computes the score.
        """
        pass

    # def __str__(self):
    #     """ Overwrite __str__ method to print information about the scores contained in the object when called."""
    #     pass

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

    def calculate(self, x, y, score_type=self.score_type):
        """
        Computes values for scores if x and y were not provided at intialization.

        Args:
            x (ndarray): (d x n) array of features.
            y (ndarray): (1 x n) array of labels.
            score_type (list or str): Should be one of: [mse, r2, adj_r2, auc, ...].
                                      If a list, then a list containing several of those entries as elements.

        Returns:
            scores (dict): Keys are score_type, values are the numeric result.
        """

def _coerce(x):
    """
    Utility function to coerce data into the correct types to be passed to sklearn

    Args:
        x (): Data to be passed to a model.


    """
    pass

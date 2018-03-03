from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pytest

class Score(object):
    """ Scoring object. Allows computation of an arbitrary score metric on an arbitrary sklearn model. """

    def __init__(self, model, score_type='mse', x=None, y=None):
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
            scores (dict): Keys are score_type, values are the numeric result.
        """
        # if score_type is None:
        #   score_type = self.score_type
        pass


class TestClass:
    def test_setters_no_data(self):
        """ Ensures that all setters are working correctly if no data is passed. """
        score_instance = Score(RFC(), 'auc')
        assert isinstance(score_instance.model, RFC)
        assert isinstance(score_instance.score_type, str)
        assert score_instance.x is None
        assert score_instance.y is None
        assert score_instance.scores is None

    def test_input_exceptions_no_data(self):
        """ Ensures that improper inputs are throwing exceptions if no data is passed. """
        with pytest.raises(TypeError):
            Score('hello', 'mse')
        with pytest.raises(TypeError):
            Score(RFC(), 5)

    def test_setters(self):
        """ Ensures that all setters are working correctly if data is passed. """
        score_instance = Score(RFC(), 'auc', np.array([[5, 5], [10, 20]]), np.array([[1, 2], [1, 2]]))
        assert isinstance(score_instance.model, RFC)
        assert isinstance(score_instance.score_type, str)
        assert isinstance(score_instance.x, np.ndarray)
        assert isinstance(score_instance.y, np.ndarray)
        assert isinstance(score_instance.scores, dict)

    def test_input_exceptions(self):
        """ Ensures that improper inputs are throwing exceptions if data is passed. """
        x, y  = load_breast_cancer(True)

        with pytest.raises(TypeError):
            Score('hello', 'mse', x, y)
        with pytest.raises(TypeError):
            Score(RFC(), 5, x, y)
        with pytest.raises(TypeError):
            Score(RFC(), 'mse', x, 'y')
        with pytest.raises(TypeError):
            Score(RFC(), 'mse', 'x', y)

        # y_prime = np.array([1, 2, 3, 4, 5, 6])
        # with pytest.raises(IndexError):
        #     Score(RFC(), 'mse', x, y_prime)
        #
        # x_prime = np.array([[1, 2], [3, 4], [5, 6]])
        # with pytest.raises(IndexError):
        #     Score(RFC(), 'mse', x_prime, y)

    def test_outputs_str(self):
        clf = RFC()
        x, y = load_breast_cancer(True)
        xt, xv, yt, yv = train_test_split(x,y, test_size=0.2, random_state=1234)

        mse = lambda y_true, y_pred: np.sum((y_true - y_pred)**2)/len(y_pred)

        clf.fit(xt, yt)
        train_pred = clf.predict(xt)
        val_pred = clf.predict(xv)
        explicit_results = [mse(i, j) for i,j in zip([yt,yv], [train_pred, val_pred])]

        score_instance = Score(RFC(), 'mse')
        score_instance.calculate(x,y)
        assert isinstance(score_instance.scores, list)
        assert np.isclose(score_instance.scores, explicit_results)

     def test_outputs_list(self):
        x, y = load_breast_cancer(True)
        xt, xv, yt, yv = train_test_split(x,y, test_size=0.2, random_state=1234)

        clf = RFC()
        clf.fit(xt, yt)
        train_pred = clf.predict(xt)
        val_pred = clf.predict(xv)

        mse = lambda y_true, y_pred : np.sum((y_true - y_pred)**2)/len(y_pred)
        acc = lambda y_true, y_pred : np.mean(y_true == y_pred)

        explicit_results = dict()
        explicit_results['mse'] = [mse(i, j) for i,j in zip([yt,yv], [train_pred, val_pred])]
        explicit_results['accuracy'] = [acc(i, j) for i,j in zip([yt,yv], [train_pred, val_pred])]

        score_instance = Score(RFC(), ['mse', 'accuracy'])
        score_instance.calculate(x,y)

        assert isinstance(score_instance.scores, dict) # Check type of output
        assert isinstance(score_instance.scores['mse'], list) # Check type of dict values
        assert isinstance(score_instance.scores['accuracy'], list) # Check type of dict values
        for key in explicit_results.keys():
            assert np.isclose(score_instance.scores[key], explicit_results[key]) # Check actual results





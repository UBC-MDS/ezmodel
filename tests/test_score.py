from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pytest



class Score(object):
    """ Scoring object. Allows computation of an arbitrary score metric on an arbitrary sklearn model. """

    def __init__(self, model, score_type='mse', x=None, y=None, random_seed=np.random.randint(1, 99999999)):
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

        self.scores_dict = {'accuracy': self._accuracy, 'mse': self._mse,
                            'auc': self._auc, "sensitivity": self._sensitivity,
                            "specificity": self._specificity, "r2": self._r2,
                            'adj_r2': self._adj_r2}

        # Type Checking for required arguments:

        # Checks for model
        if "sklearn" not in str(type(model)):
            raise TypeError("Model must be an sklearn classifier or regression object.")

        # Checks for score_type
        if isinstance(score_type, str):
            if score_type not in self.scores_dict.keys():
                raise TypeError("{} is not a supported score function. Please try one of: {}".format(score_type, ", ".join(self.scores_dict.keys())))

        elif isinstance(score_type, list):
            for i in score_type:
                if not isinstance(i, str) or i not in self.scores_dict.keys():
                    raise TypeError("{} is not a valid input to scores_type. Please try one of: {}".format(i, ", ".join(self.scores_dict.keys())))
        else:
            raise TypeError("score_type should be a string or a list")

        # Check for random_seed
        if not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer.")

        # Check for x and y inputs
        if x is not None:
            if y is None:
                raise TypeError("y must also be supplied if x is supplied.")
            else:  # Set attributes and compute score for the given data.
                self.model = model
                self.x = x
                self.y = y
                self.score_type = score_type
                self.random_seed = random_seed
                self.scores = self.calculate(self.x, self.y, self.score_type)

        elif y is not None:  # This will always raise, since we will only get here if x is None
            raise TypeError("x must also be supplied if y is supplied.")

        else: # If no x and y are passed, set attributes and finish __init__()
            self.model = model
            self.score_type = score_type
            self.random_seed = random_seed
            self.scores = None

    def __str__(self):
        """ Overwrite __str__ method to print information about the scores contained in the object when called."""
        if self.x is not None:
            if isinstance(self.score_type, list):
                return "Score object for: \nModel Type: {} \nScores: {}".format(type(self.model),
                                                                          [i for i in zip(self.score_type,
                                                                                          self.scores)])
            else:
                return "Score object for: \nModel Type: {} \nScore: {}={}".format(type(self.model),
                                                                            self.score_type,
                                                                            self.scores)

        else:
            if isinstance(self.score_type, list):
                return "Score object for: \nModel Type: {} \nScore Types: {}".format(type(self.model),
                                                                                     self.score_type)
            else:
                return "Score object for: \nModel Type: {} \nScore Type: {}".format(type(self.model),
                                                                                    self.score_type)

    def _splitfitnpredict(self):
        """
        Hidden method to split the data in the Score() object, to fit the model, and then to predict labels.

        Returns:
            Two lists. The first contains the true split labels, the second contains the predicted labels.
        """
        xt, xv, yt, yv = train_test_split(self.x, self.y, random_state=self.random_seed)

        self.model.fit(xt, yt)
        t_pred = self.model.predict(xt)
        v_pred = self.model.predict(xv)

        return [yt, yv], [t_pred, v_pred]

    def _accuracy(self):
        """ Computes Accuracy of a model. Number of correct predictions over total number of predictions.
            Uses self.model, self.x and self.y """
        t_labels, p_labels = self._splitfitnpredict()

        scores = [np.mean(t_labels[0] == p_labels[0]),
                  np.mean(t_labels[1] == p_labels[1])]

        return scores

    def _mse(self):
        """ Computes Mean Squared Error. Uses self.model, self.x and self.y"""
        t_labels, p_labels = self._splitfitnpredict()

        scores = [np.sum((t_labels[0] - p_labels[0])**2)/len(t_labels[0]),
                  np.sum((t_labels[1] - p_labels[1])**2)/len(t_labels[1])]

        return scores

    def _r2(self):
        """ Computes R-Squared. Uses self.model, self.x and self.y """
        t_labels, p_labels = self._splitfitnpredict()

        scores = [(1 - ((np.sum((t_labels[0] - p_labels)**2))/(np.sum((t_labels[0] - np.mean(t_labels[0]))**2)))),
                  (1 - ((np.sum((t_labels[1] - p_labels)**2))/(np.sum((t_labels[1] - np.mean(t_labels[1]))**2))))]

        return scores

    def _adj_r2(self):
        """ Computes Adjusted R-Squared. Uses self.model, self.x and self.y """
        r2s = self._r2

        scores = [(1 - r2s[0]*((self.x.shape[0] - 1)/(self.x.shape[0] - self.x.shape[1] - 1))),
                  (1 - r2s[1]*((self.x.shape[0] - 1)/(self.x.shape[0] - self.x.shape[1] - 1)))]

        return scores

    def _auc(self):
        """ Computes Area Under the Receiver Operator Curve. Uses self.model, self.x and self.y"""

        sens = self._sensitivity()
        spec = self._specificity()
        # raise NotImplementedError("A general function for AUC is harder than expected! Coming Soon.")
        pass
    def _sensitivity(self):
        """
        Computes model sensitivity. Used for AUC. Uses self.model, self.x and self.y

        Equal to TP/(TP + FN)
        """
        t_labels, p_labels = self._splitfitnpredict()

        scores = [self._truepos(t_labels[0], p_labels[0])/(self._truepos(t_labels[0], p_labels[0]) +
                                                           self._falseneg(t_labels[0], p_labels[0])),

                  self._truepos(t_labels[1], p_labels[1])/(self._truepos(t_labels[1], p_labels[1]) +
                                                           self._falseneg(t_labels[1], p_labels[1]))]
        return scores

    def _specificity(self):
        """
        Computes model specificity. Used for AUC. Uses self.model, self.x and self.y

        Equal to TN/(TN + FP)
        """
        t_labels, p_labels = self._splitfitnpredict()

        scores = [self._trueneg(t_labels[0], p_labels[0])/(self._trueneg(t_labels[0], p_labels[0]) +
                                                           self._falsepos(t_labels[0], p_labels[0])),

                  self._trueneg(t_labels[1], p_labels[1])/(self._trueneg(t_labels[1], p_labels[1]) +
                                                           self._falsepos(t_labels[1], p_labels[1]))]
        return scores

    def _truepos(self, y_true, y_pred):
        """ Computes the number of true positives in a set of predictions """
        return sum([1 for i in range(len(y_true)) if y_true[i].all() == 1 and y_pred[i].all() == 1])

    def _falsepos(self, y_true, y_pred):
        """ Computes the number of false positives in a set of predictions """
        return sum([1 for i in range(len(y_true)) if y_true[i].all() == 0 and y_pred[i].all() == 1])

    def _falseneg(self, y_true, y_pred):
        """ Computes the number of true negatives in a set of predictions """
        return sum([1 for i in range(len(y_true)) if y_true[i].all() == 1 and y_pred[i].all() == 0])

    def _trueneg(self, y_true, y_pred):
        """ Computes the number of false negatives in a set of predictions """
        return sum([1 for i in range(len(y_true)) if y_true[i].all() == 0 and y_pred[i].all() == 0])

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
        # Type Checking:
        self.x = _coerce(x)
        self.y = _coerce(y)

        if score_type is None:
            score_type = self.score_type

        if isinstance(score_type, str):
            try:
                scores = self.scores_dict[score_type]()
                self.scores = scores
                return self.scores

            except KeyError:
                print("{} is not a supported score function. Please try one of: {}".format(score_type, ", ".join(self.scores_dict.keys())))

        elif isinstance(score_type, list):
            scores = dict()
            for i in score_type:
                try:
                    c_score = self.scores_dict[i]()
                    scores[i] = c_score
                except KeyError:
                    print("{} is not a supported score function. Please try one of: {}".format(i, ", ".join(self.scores_dict.keys())))
                    scores[i] = None
            self.scores = scores
            return scores

        else:
            return TypeError("score_type must be a list or string")


def _coerce(x):
    """
    Utility function to coerce data into the correct types to be passed to sklearn

    Args:
        x (??): Data to be passed to a model.

    Returns:
        np.ndarray containing the data from x

    Notes:
        Works for pandas DataFrames and nested lists currently. Investigating other types that need coercing.
    """
    if isinstance(x, pd.DataFrame):
        return pd.as_matrix(x)
    elif isinstance(x, list):
        return np.asarray(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError("{} is currently not supported. Please input a numpy array, pandas DataFrame, or nested list".format(type(x)))
    else:
        return x


class TestClass:
    def test_setters_no_data(self):
        """ Ensures that all setters are working correctly if no data is passed. """
        score_instance = Score(RFC(), 'auc')
        assert isinstance(score_instance.model, RFC)
        assert isinstance(score_instance.score_type, str)
        try:
            assert score_instance.x is None
        except AttributeError:
            pass
        try:
            assert score_instance.y is None
        except AttributeError:
            pass
        try:
            assert score_instance.scores is None
        except AttributeError:
            pass

    def test_input_exceptions_no_data(self):
        """ Ensures that improper inputs are throwing exceptions if no data is passed. """
        with pytest.raises(TypeError):
            Score('hello', 'mse')
        with pytest.raises(TypeError):
            Score(RFC(), 5)

    def test_setters(self):
        """ Ensures that all setters are working correctly if data is passed. """
        score_instance = Score(RFC(), 'sensitivity', np.array([[5, 5], [10, 20]]), np.array([[1, 2], [1, 2]]))
        assert isinstance(score_instance.model, RFC)
        assert isinstance(score_instance.score_type, str)
        assert isinstance(score_instance.x, np.ndarray)
        assert isinstance(score_instance.y, np.ndarray)
        assert isinstance(score_instance.scores, (list, dict))

    def test_input_exceptions(self):
        """ Ensures that improper inputs are throwing exceptions if data is passed. """
        x, y = load_breast_cancer(True)

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
        explicit_results = [mse(i, j) for i, j in zip([yt, yv], [train_pred, val_pred])]

        score_instance = Score(RFC(), 'mse', random_seed=1234)
        score_instance.calculate(x, y)
        assert isinstance(score_instance.scores, list)
        assert np.isclose(all(score_instance.scores), all(explicit_results))

    def test_outputs_list(self):
        x, y = load_breast_cancer(True)
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=1234)

        clf = RFC()
        clf.fit(xt, yt)
        train_pred = clf.predict(xt)
        val_pred = clf.predict(xv)

        mse = lambda y_true, y_pred: np.sum((y_true - y_pred)**2)/len(y_pred)
        acc = lambda y_true, y_pred: np.mean(y_true == y_pred)

        explicit_results = dict()
        explicit_results['mse'] = [mse(i, j) for i, j in zip([yt,yv], [train_pred, val_pred])]
        explicit_results['accuracy'] = [acc(i, j) for i, j in zip([yt,yv], [train_pred, val_pred])]

        score_instance = Score(RFC(), ['mse', 'accuracy'], random_seed=1234)
        score_instance.calculate(x, y)

        assert isinstance(score_instance.scores, dict) # Check type of output
        assert isinstance(score_instance.scores['mse'], list) # Check type of dict values
        assert isinstance(score_instance.scores['accuracy'], list) # Check type of dict values
        for key in explicit_results.keys():
            assert np.isclose(all(score_instance.scores[key]), all(explicit_results[key])) # Check actual results





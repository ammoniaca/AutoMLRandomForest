from dto.dto import RequestDTO
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_estimator(request: RequestDTO):
    switcher = {
        "classification": RandomForestClassifier(),
        "regression": RandomForestRegressor()
    }
    return switcher.get(request.task.value, None)


def is_classifier(estimator):
    """Return True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.

    Examples
    --------
    >>> from sklearn.base import is_classifier
    >>> from sklearn.svm import SVC, SVR
    >>> classifier = SVC()
    >>> regressor = SVR()
    >>> is_classifier(classifier)
    True
    >>> is_classifier(regressor)
    False
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Return True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.

    Examples
    --------
    >>> from sklearn.base import is_regressor
    >>> from sklearn.svm import SVC, SVR
    >>> classifier = SVC()
    >>> regressor = SVR()
    >>> is_regressor(classifier)
    False
    >>> is_regressor(regressor)
    True
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"

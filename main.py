# to create synthetic data for regression (by sklearn)
from sklearn.datasets import make_regression

# to split data in train and test subsets
from sklearn.model_selection import train_test_split

from dto.dto import RequestDTO
from enumerator.estimator_task import EstimatorTask
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer
from enumerator.scores import RegressionScore

# import AutoML
from automl_rf import AutoMLRandomForest

if __name__ == '__main__':

    # Create synthetic data for regression (by sklearn)
    X, y = make_regression(
        n_samples=500,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=True
    )

    # Split arrays or matrices into random train and test subsets (by sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    space_rf = {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'bootstrap': [True, False],
        'max_depth': [10, 20],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 10],
        'n_estimators': [10, 100]  # [x for x in range(10, 1000, 10)]
    }

    # AutoML - RF regression: example for GridSearchCV
    request_grid_search = RequestDTO(
        task=EstimatorTask.REGRESSION,
        space=space_rf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimizer=HyperparameterOptimizer.GRID_SEARCH_CV,
        cv=5,  # (default=5). If you wish LOO then cv=len(X_train) or cv=len(y_train)
        verbose=2,  # (default=0 -> no display for better performance)
        tuning_scoring=RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR,
        validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR, RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
    )

    result_grid_search: dict = AutoMLRandomForest(request=request_grid_search).run()
    for key, value in result_grid_search.items():
        print(f"{key}: {value}")
    print("")

    # AutoML - RF regression : example for RandomizedSearchCV
    request_randomized_search = RequestDTO(
        task=EstimatorTask.REGRESSION,
        space=space_rf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimizer=HyperparameterOptimizer.RANDOMIZED_SEARCH_CV,
        cv=5,  # (default=5). If you wish LOO then cv=len(X_train) or cv=len(y_train)
        verbose=2,  # (default=0 -> no display for better performance)
        n_iter=100,  # for RandomizedSearchCV: Number of sampled. n_iter trades off runtime vs quality of the solution.
        tuning_scoring=RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR,
        validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR, RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
    )

    result_randomized_search: dict = AutoMLRandomForest(request=request_randomized_search).run()
    for key, value in result_randomized_search.items():
        print(f"{key}: {value}")
    print("")


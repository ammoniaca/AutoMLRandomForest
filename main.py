from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from dto.dto import RequestDTO
from enumerator.estimator_task import EstimatorTask
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer
from enumerator.scores import RegressionScore
from tuning.gridsearchcv_concrete_strategy import GridSearchCVStrategy

if __name__ == '__main__':
    # Create synthetic for regression (by sklearn)
    X, y = make_regression(
        n_samples=500,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=True
    )

    # Split arrays or matrices into random train and test subsets (by sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # AutoML - RF: example for GridSearchCV

    request = RequestDTO(
        task=EstimatorTask.REGRESSION,
        space={
            'bootstrap': [True, False],
            'max_depth': [10, 20],
            'max_features': ['sqrt'],
            'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 10],
            'n_estimators': [130, 250]
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimizer=HyperparameterOptimizer.GRID_SEARCH_CV,
        cv=5,  # if you wish LOO then cv=len(X_train) or cv=len(y_train)
        verbose=2,
        tuning_scoring=RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR,
        validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR, RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
    )

    grid_search_result = GridSearchCVStrategy(request=request).optimize()

    # AutoML - RF: example for RandomizedSearchCV
# to create synthetic data for regression (by sklearn)
from dataclasses import dataclass, field, asdict

from sklearn.datasets import make_regression

# to split data in train and test subsets
from sklearn.model_selection import train_test_split

# save trained model
import pickle
import pandas as pd
from datetime import datetime

from dto.space_rf_dto import SpaceRandomForest
from dto.dto import RequestDTO
from enumerator.estimator_task import EstimatorTask
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer
from enumerator.scores import RegressionScore

# import AutoML
from automl_rf import AutoMLRandomForest

import optuna
from optuna.distributions import IntDistribution, CategoricalDistribution, FloatDistribution

if __name__ == '__main__':
    # Create synthetic data for regression (by sklearn)
    X, y = make_regression(
        n_samples=500,
        n_features=4,
        n_informative=2,
        random_state=42,
        shuffle=True
    )

    # Split arrays or matrices into random train and test subsets (by sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # START of EXERIMENT

    # setting
    save = True
    test_grid_search_cv = False
    test_randomized_search_cv = False
    test_optuna = True
    # saving
    gridsearch_save_label = "gridsearch_rf"
    randomizedsearch_save_label = "randomizedsearch_rf"
    optuna_save_label = "optuna_rf"

    # vanilla (Hyperparametrization need to more accurate of vanilla setting)
    vanilla_space_rf = SpaceRandomForest(
        n_estimators=[100],
        criterion=['squared_error'],
        max_depth=[None],
        min_samples_split=[2],
        min_samples_leaf=[1],
        min_weight_fraction_leaf=[0.0],
        max_features=[1.0],
        max_leaf_nodes=[None],
        min_impurity_decrease=[0.0],
        bootstrap=[True],
        oob_score=[False],
        warm_start=[False],
        ccp_alpha=[0.0],
        max_samples=[None]
    )

    # for GridSearchCV and RandomizedSearchCV
    space_rf = SpaceRandomForest(
        n_estimators=[10, 100],  # [x for x in range(10, 1000, 10)]
        criterion=['absolute_error', 'friedman_mse', 'squared_error'],
        max_depth=[10, 20, 30],
        min_samples_split=[1, 2, 3],
        min_samples_leaf=[11, 22, 33],
        min_weight_fraction_leaf=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        max_features=['sqrt', 'log2', None],
        max_leaf_nodes=[1, 12, 42],
        min_impurity_decrease=[0.0, 0.1, 0.3]
    )
    print("")

    # AutoML - RF regression: example for GridSearchCV
    if test_grid_search_cv:
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
            validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR,
                                RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
        )

        result_grid_search: dict = AutoMLRandomForest(request=request_grid_search).run()
        for key, value in result_grid_search.items():
            print(f"{key}: {value}")
        print("")

        if save:
            # datetime object containing current date and time
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%dT%H%M%S")

            # Save best model parameters
            df1 = pd.DataFrame(result_grid_search.get('best_params'), index=[0])
            df1.to_csv(f"{gridsearch_save_label}_best_param_{dt_string}.csv", index=False)

            # Save y_true and y_pred from best model
            data = {"y_true": y_test.tolist(), "y_pred": result_grid_search.get("y_pred").tolist()}
            df2 = pd.DataFrame.from_dict(data)
            df2.to_csv(f"{gridsearch_save_label}_prediction_{dt_string}.csv", index=False)

            # Save scores
            scores = result_grid_search.get("score")
            df3 = pd.DataFrame(scores, index=[0])
            df3.to_csv(f"{gridsearch_save_label}_metric_score_{dt_string}.csv", index=False)
            print("")

    # AutoML - RF regression : example for RandomizedSearchCV
    if test_randomized_search_cv:
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
            n_iter=100,
            # try 1000!!! # RandomizedSearchCV: Number of sampled. n_iter trades off runtime vs quality of the solution.
            tuning_scoring=RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR,
            validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR,
                                RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
        )

        result_randomized_search: dict = AutoMLRandomForest(request=request_randomized_search).run()
        for key, value in result_randomized_search.items():
            print(f"{key}: {value}")
        print("")

        if save:
            # datetime object containing current date and time
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%dT%H%M%S")

            # Save best model parameters
            df1 = pd.DataFrame(result_randomized_search.get('best_params'), index=[0])
            df1.to_csv(f"{randomizedsearch_save_label}_best_param_{dt_string}.csv", index=False)

            # Save y_true and y_pred from best model
            data = {"y_true": y_test.tolist(), "y_pred": result_randomized_search.get("y_pred").tolist()}
            df2 = pd.DataFrame.from_dict(data)
            df2.to_csv(f"{randomizedsearch_save_label}_prediction_{dt_string}.csv", index=False)

            # Save scores
            scores = result_randomized_search.get("score")
            df3 = pd.DataFrame(scores, index=[0])
            df3.to_csv(f"{randomizedsearch_save_label}_metric_score_{dt_string}.csv", index=False)
            print("")

    if test_optuna:
        # AutoML - RF regression: example for Optuna

        space_optuna_rf = SpaceRandomForest(
            n_estimators=IntDistribution(low=100, high=2000, step=10),
            criterion=CategoricalDistribution(['absolute_error', 'friedman_mse', 'squared_error']),
            max_depth=IntDistribution(low=10, high=100, step=1),
            min_samples_split=IntDistribution(low=1, high=100, step=1),
            min_samples_leaf=IntDistribution(low=0, high=100, step=1),
            min_weight_fraction_leaf=FloatDistribution(low=0.0, high=0.5, step=0.01),
            max_features=CategoricalDistribution(['sqrt', 'log2', None]),
            max_leaf_nodes=IntDistribution(low=1, high=100, step=1),
            min_impurity_decrease=FloatDistribution(low=0.0, high=1.0, step=0.1)
        )
        print("")

        request_optuna = RequestDTO(
            task=EstimatorTask.REGRESSION,
            space=space_optuna_rf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            optimizer=HyperparameterOptimizer.OPTUNA_CV,
            cv=5,  # (default=5). If you wish LOO then cv=len(X_train) or cv=len(y_train)
            n_iter=10,  # i.e. number of trial for Optuna # try 1000!!!
            verbose=2,
            tuning_scoring=RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR,
            validation_scoring=[RegressionScore.R2, RegressionScore.MAX_ERROR,
                                RegressionScore.NEG_ROOT_MEAN_SQUARED_ERROR]
        )
        result_optuna: dict = AutoMLRandomForest(request=request_optuna).run()
        for key, value in result_optuna.items():
            print(f"{key}: {value}")
        print("")

        if save:
            # datetime object containing current date and time
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%dT%H%M%S")

            # Save best model parameters
            df1 = pd.DataFrame(result_optuna.get('best_params'), index=[0])
            df1.to_csv(f"{optuna_save_label}_best_param_{dt_string}.csv", index=False)

            # Save y_true and y_pred from best model
            data = {"y_true": y_test.tolist(), "y_pred": result_optuna.get("y_pred").tolist()}
            df2 = pd.DataFrame.from_dict(data)
            df2.to_csv(f"{optuna_save_label}_prediction_{dt_string}.csv", index=False)

            # Save scores
            scores = result_optuna.get("score")
            df3 = pd.DataFrame(scores, index=[0])
            df3.to_csv(f"{optuna_save_label}_metric_score_{dt_string}.csv", index=False)
            print("")
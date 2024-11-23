import optuna
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.optimizer_strategy import IOptimizerStrategy
from tuning.estimator_type import get_estimator, is_regressor
import numpy as np


class OptunaCVStrategy(IOptimizerStrategy):

    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self) -> ResponseTuningDTO:
        """
        Run the optimization process.
        """
        estimator = get_estimator(self._request)
        optuna_search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=self._request.space.to_dict(),
            cv=self._request.cv,
            scoring=self._request.tuning_scoring.value,
            n_trials=self._request.n_iter,
            verbose=self._request.verbose
        )

        optuna_search.fit(
            X=np.array(self._request.X_train, dtype="float64"),
            y=np.array(self._request.y_train, dtype="float64")
        )

        # print("Best trial:")
        # trial = optuna_search.study_.best_trial
        #
        # print("  Value: ", trial.value)
        # print("  Params: ")
        # for key, value in trial.params.items():
        #     print("    {}: {}".format(key, value))
        # # trial.params
        # print("")

        trial = optuna_search.study_.best_trial

        if is_regressor(estimator):
            best_estimator = RandomForestRegressor(**trial.params).fit(self._request.X_train, self._request.y_train)
        else:
            best_estimator = RandomForestClassifier(**trial.params).fit(self._request.X_train, self._request.y_train)

        return ResponseTuningDTO(
            strategy=self._request.optimizer.value,
            best_estimator=best_estimator,
            best_params=trial.params,
            tuning_metric={f'tuning_{self._request.tuning_scoring.value}': trial.value}
        )

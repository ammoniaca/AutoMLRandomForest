from sklearn.model_selection import GridSearchCV

from tuning.estimator_type import get_estimator
from tuning.optimizer_strategy import IOptimizerStrategy
from dto.dto import RequestDTO, ResponseTuningDTO


class GridSearchCVStrategy(IOptimizerStrategy):

    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self) -> ResponseTuningDTO:
        """
         hyperparameter tuning by exhaustively searching through a predefined grid of
         parameter combinations, evaluating each combination using cross-validation,
         and providing us with the best set of parameters that maximize the model’s performance

        :return: ResponseTuningDTO
        """
        grid_search = GridSearchCV(
            estimator=get_estimator(self._request),
            param_grid=self._request.space,
            cv=self._request.cv,
            scoring=self._request.tuning_scoring.value,
            verbose=self._request.verbose,
            pre_dispatch=self._request.n_jobs
        )
        grid_search.fit(self._request.X_train, self._request.y_train)
        return ResponseTuningDTO(
            best_estimator=grid_search.best_estimator_,
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_
        )

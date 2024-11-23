from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.grid_search_cv_concrete_strategy import GridSearchCVStrategy
from tuning.randomized_search_cv_concrete_strategy import RandomizedSearchCVStrategy
from tuning.optuna_concrete_strategy import OptunaCVStrategy


class Optimizer:
    def __init__(self, request: RequestDTO):
        self._request: RequestDTO = request
        self._strategy: dict = {
            "gridSearchCV": GridSearchCVStrategy(self._request),
            "randomizedSearchCV": RandomizedSearchCVStrategy(self._request),
            "optunaCV": OptunaCVStrategy(self._request),
        }

    def optimize(self) -> ResponseTuningDTO:
        return self._strategy.get(self._request.optimizer.value).optimize()



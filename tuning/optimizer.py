from dto.dto import RequestDTO, _ResponseTuningDTO
from tuning.grid_search_cv_concrete_strategy import GridSearchCVStrategy
from tuning.randomized_search_cv_concrete_strategy import RandomizedSearchCVStrategy
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer




class Optimizer:
    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self) -> _ResponseTuningDTO:
        _strategy: dict = {
            "gridSearchCV": GridSearchCVStrategy(self._request).optimize(),
            "randomizedSearchCV": GridSearchCVStrategy(self._request).optimize()
        }
        return _strategy.get(self._request.optimizer.value)



from dto.dto import RequestDTO, ResponseTuningDTO
from gridsearchcv_concrete_strategy import GridSearchCVStrategy
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer


class Optimizer:
    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self):
        if self._request.optimizer == HyperparameterOptimizer.GRID_SEARCH_CV:
            response_tuning: ResponseTuningDTO = GridSearchCVStrategy(self._request)
        elif self._request.optimizer == HyperparameterOptimizer.RANDOMIZED_SEARCH_CV:
            pass

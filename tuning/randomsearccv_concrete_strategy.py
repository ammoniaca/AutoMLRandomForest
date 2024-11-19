from sklearn.model_selection import RandomizedSearchCV

from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.optimizer_strategy import IOptimizerStrategy


class RandomizedCVStrategy(IOptimizerStrategy):

    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self) -> ResponseTuningDTO:
        pass
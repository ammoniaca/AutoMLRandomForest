from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.optimizer import Optimizer


class AutoMLRandomForest:

    def __init__(self, request: RequestDTO):
        self._request = request

    def solver(self):
        response: ResponseTuningDTO = Optimizer(self._request)

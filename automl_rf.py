from dataclasses import dataclass, asdict

from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.optimizer import Optimizer
from validator.model_validation import ModelValidator
from numpy.typing import ArrayLike


class AutoMLRandomForest:

    def __init__(self, request: RequestDTO):
        self._request = request

    def run(self) -> dict:
        response: ResponseTuningDTO = Optimizer(self._request).optimize()
        model_validator = ModelValidator(request=self._request, response=response)
        scores: dict = model_validator.get_scores()
        y_pred: dict = model_validator.get_predict()
        # combine dicts
        return asdict(response) | scores | y_pred





from dataclasses import asdict

from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.optimizer import Optimizer
from validator.model_validation import ModelValidator


class AutoMLRandomForest:

    def __init__(self, request: RequestDTO):
        self._request = request

    def run(self) -> dict:
        response: ResponseTuningDTO = Optimizer(self._request).optimize()
        model_validator = ModelValidator(request=self._request, response=response)
        scores: dict = model_validator.get_scores()
        scores.update(response.tuning_metric)
        y_pred: dict = model_validator.get_predict()
        response_dict = asdict(response)
        # combine dicts
        return response_dict | {"score": scores} | y_pred





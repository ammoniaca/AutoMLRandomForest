from dto.dto import RequestDTO, ResponseTuningDTO
from sklearn.metrics import get_scorer


class ModelValidator:

    def __init__(self, request: RequestDTO, response: ResponseTuningDTO):
        self._request = request
        self._response = response
        self._y_pred = self._response.best_estimator.predict(self._request.X_test)

    def get_predict(self) -> dict:
        return {"y_pred": self._y_pred}

    def get_scores(self) -> dict:
        scores_result = {}
        for s in self._request.validation_scoring:
            scorer = get_scorer(s.value)
            scores_result[s.value] = scorer(
                self._response.best_estimator,
                self._request.X_test,
                self._request.y_test
            )
        return scores_result

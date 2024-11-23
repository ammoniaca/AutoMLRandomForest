from sklearn.model_selection import RandomizedSearchCV

from dto.dto import RequestDTO, ResponseTuningDTO
from tuning.estimator_type import get_estimator
from tuning.optimizer_strategy import IOptimizerStrategy


class RandomizedSearchCVStrategy(IOptimizerStrategy):

    def __init__(self, request: RequestDTO):
        self._request = request

    def optimize(self) -> ResponseTuningDTO:
        """
        Run the optimization process.
        """
        randomized_search = RandomizedSearchCV(
            estimator=get_estimator(self._request),
            param_distributions=self._request.space.to_dict(),
            n_iter=self._request.n_iter,
            scoring=self._request.tuning_scoring.value,
            n_jobs=self._request.n_jobs,
            cv=self._request.cv,
            verbose=self._request.verbose,
            random_state=self._request.seed,
            return_train_score=True

        )
        randomized_search.fit(self._request.X_train, self._request.y_train)
        return ResponseTuningDTO(
            strategy=self._request.optimizer.value,
            best_estimator=randomized_search.best_estimator_,
            best_params=randomized_search.best_params_,
            tuning_metric={f'tuning_{self._request.tuning_scoring.value}': randomized_search.best_score_}
        )

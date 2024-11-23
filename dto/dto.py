from dataclasses import dataclass, field, asdict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from numpy.typing import ArrayLike
from typing import Union

from dto.space_rf_dto import SpaceRandomForest
from enumerator.scores import ClassificationScore, RegressionScore
from enumerator.estimator_task import EstimatorTask
from enumerator.hyperparameter_optimizers import HyperparameterOptimizer


@dataclass
class DataAnalysisDTO:
    X_train: ArrayLike
    X_test: ArrayLike
    y_train: ArrayLike
    y_test: ArrayLike


@dataclass
class RequestDTO:
    """
    """
    task: EstimatorTask
    X_train: ArrayLike
    y_train: ArrayLike
    X_test: ArrayLike
    y_test: ArrayLike
    optimizer: HyperparameterOptimizer
    tuning_scoring: ClassificationScore | RegressionScore
    validation_scoring: list[ClassificationScore] | list[RegressionScore]
    space: SpaceRandomForest = field(repr=False, default=None)
    cv: int = field(repr=False, default=5)
    verbose: int = field(repr=False, default=0)
    n_iter: int = field(repr=False, default=10)
    n_jobs: int = field(repr=False, default=1)
    seed: int = field(repr=False, default=None)

    def __post_init__(self):

        if self.verbose < 0:
            raise ValueError(f"verbose {self.verbose} is not supported. Supported values are >= 0")

        # TODO: check tuning_scoring: [ClassificationScore] | [RegressionScore]

        # TODO: check validation_scoring: [ClassificationScore] | [RegressionScore]

        # TODO: check if scores are valid for classification

        # TODO: check if scores are valid for regression

        # verify if task is classification or regression, otherwise raise an error
        if self.task not in [EstimatorTask.CLASSIFICATION, EstimatorTask.REGRESSION]:
            raise ValueError(f"task {self.task} is not supported")

        # verify is the number of data features points for model training (X_train) is equal to labels points
        # for model training (y_train), otherwise raise an error
        if len(self.X_train) != len(self.y_train):
            raise ValueError(f"The number of elements of X_train: {len(self.X_train)} "
                             f"differs from the number of elements of y_train: {len(self.y_train)}")

        # verify is the number of data features points for model validation (X_test) is equal to labels points
        # for model validation (y_test), otherwise raise an error
        if len(self.X_train) != len(self.y_train):
            raise ValueError(f"The number of elements of X_train: {len(self.X_train)} "
                             f"differs from the number of elements of y_train: {len(self.y_train)}")

        # verify if cv number is between 1 or the total number of examples N in the training dataset
        # is cv = N we are using the leave-one-out cross-validation (LOOCV), otherwise raise an error
        if not 1 <= self.cv < len(self.X_train):
            raise ValueError("the value of cv must be between 1 and the maximum number of training elements")

    def space_to_dict(self):
        # convert space object in a dictionary removing None value
        return {k: v for k, v in asdict(self.space).items() if v is not None}



@dataclass
class ResponseTuningDTO:
    """
    """
    strategy: str
    best_estimator: Union[
        RandomForestClassifier,
        RandomForestRegressor,
        RandomForestClassifier(),
        RandomForestRegressor()
    ]
    best_params: dict
    tuning_metric: dict

from enum import Enum


class HyperparameterOptimizer(Enum):
    RANDOMIZED_SEARCH_CV = "randomizedSearchCV"
    GRID_SEARCH_CV = "gridSearchCV"

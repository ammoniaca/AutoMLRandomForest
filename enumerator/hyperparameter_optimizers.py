from enum import Enum


class HyperparameterOptimizer(Enum):
    RANDOMIZED_SEARCH_CV = "randomizedsearchcv"
    GRID_SEARCH_CV = "gridSearchcv"

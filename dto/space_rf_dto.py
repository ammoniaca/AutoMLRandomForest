from dataclasses import dataclass, field, asdict
import optuna
from typing import Union

@dataclass
class SpaceRandomForest:
    """

    """
    n_estimators: Union[int, list[int], optuna.distributions] = field(repr=False, default=None)
    criterion: Union[str, list[str], optuna.distributions] = field(repr=False, default=None)
    max_depth: Union[int, list[int], optuna.distributions] = field(repr=False, default=None)
    min_samples_split: Union[int, float, list[int], list[float], optuna.distributions] = field(repr=False, default=None)
    min_samples_leaf: Union[int, float, list[int], list[float], optuna.distributions] = field(repr=False, default=None)
    min_weight_fraction_leaf: Union[float, list[float], optuna.distributions] = field(repr=False, default=None)
    max_features: Union[int, float, str, list[int], list[float], list[str], optuna.distributions] = field(repr=False, default=None)
    max_leaf_nodes: Union[int, list[int], optuna.distributions] = field(repr=False, default=None)
    min_impurity_decrease: Union[float, list[float], optuna.distributions] = field(repr=False, default=None)
    bootstrap: Union[bool, list[bool], optuna.distributions] = field(repr=False, default=None)
    oob_score: Union[bool, list[bool], optuna.distributions] = field(repr=False, default=None)
    warm_start: Union[bool, optuna.distributions] = field(repr=False, default=None)
    ccp_alpha: Union[float, list[float, optuna.distributions]] = field(repr=False, default=None)
    max_samples: Union[int, float, None, list[int], list[float], optuna.distributions] = field(repr=False, default=None)

    def to_dict(self):
        # convert space object in a dictionary removing None value
        return {k: v for k, v in asdict(self).items() if v is not None}

from abc import ABC, abstractmethod

from dto.dto import _ResponseTuningDTO


class IOptimizerStrategy(ABC):
    @abstractmethod
    def optimize(self) -> _ResponseTuningDTO:
        pass

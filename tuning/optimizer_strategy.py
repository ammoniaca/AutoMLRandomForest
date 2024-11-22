from abc import ABC, abstractmethod

from dto.dto import ResponseTuningDTO


class IOptimizerStrategy(ABC):
    @abstractmethod
    def optimize(self) -> ResponseTuningDTO:
        pass

from abc import ABC, abstractmethod
from typing import List, Tuple


class ModelInterface(ABC):
    """ Basic Interface for external models to interface with test harness """

    def __init__(self, model_name: str):
        self.name = model_name

    @abstractmethod
    def predict(self,
                text1: str,
                text2: str) -> Tuple[int, float]:
        raise NotImplementedError("Implement a predict method.")


    @abstractmethod
    def predict_batch(self,
                      inputs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
        raise NotImplementedError("Implement predict batch")

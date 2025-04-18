from abc import ABC, abstractmethod
from typing import Any, List
import random
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Distribution(ABC):
    low: float
    high: float

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
    
    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    @abstractmethod
    def discretize(self, n: float) -> Any:
        return n


class FloatDistribution(Distribution):
    def __init__(self, low: float, high: float):
        super().__init__(low, high)

    def discretize(self, n: float) -> float:
        return n


class IntDistribution(Distribution):
    def __init__(self, low: int, high: int):
        super().__init__(float(low - 0.5), float(high + 0.5))

    def discretize(self, n: float) -> int:
        return round(n)


class ChoiceDistribution(Distribution):
    def __init__(self, choices: List[Any]):
        self.choices = choices
        super().__init__(0, len(choices) - 1)

    def discretize(self, n: float) -> Any:
        return self.choices[round(n)]
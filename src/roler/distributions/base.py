from abc import ABC, abstractmethod
from typing import Any

class Distribution(ABC):
    """
    Abstract base class for probability distributions.
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        Samples a value from the distribution.

        Returns:
            Any: A sample from the distribution.
        """
        pass

class NumberDistribution(Distribution):
    @abstractmethod
    def sample(self) -> float:
        pass
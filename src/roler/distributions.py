from abc import ABC, abstractmethod
from typing import Any, List

class Distribution(ABC):
    """
    Abstract base class for distributions.
    Defines the basic structure and enforces the implementation of a discretize method.
    """
    def __init__(self, low: float, high: float):
        """
        Initializes a distribution with a lower and upper bound.

        Args:
            low (float): The lower bound of the distribution.
            high (float): The upper bound of the distribution.

        Raises:
            TypeError: If low or high are not numeric (int or float).
            ValueError: If low is not less than high.
        """
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError("low and high must be numeric (int or float)")
        if low >= high:
            raise ValueError("low must be less than high")

        self.low = float(low)
        self.high = float(high)

    @abstractmethod
    def discretize(self, n: float) -> Any:
        """
        Abstract method to discretize a value.  Must be implemented by subclasses.

        Args:
            n (float): The value to discretize.

        Returns:
            Any: The discretized value.
        """
        return n


class FloatDistribution(Distribution):
    """
    A distribution that returns a float.
    """
    def __init__(self, low: float, high: float):
        """
        Initializes a FloatDistribution with a lower and upper bound.

        Args:
            low (float): The lower bound of the distribution.
            high (float): The upper bound of the distribution.
        """
        super().__init__(low, high)

    def discretize(self, n: float) -> float:
        """
        Discretizes a float value (returns the same float).

        Args:
            n (float): The value to discretize.

        Returns:
            float: The discretized value (same as input).
        """
        return n


class IntDistribution(Distribution):
    """
    A distribution that returns an integer.
    """
    def __init__(self, low: int, high: int):
        """
        Initializes an IntDistribution with a lower and upper bound.

        Args:
            low (int): The lower bound of the distribution.
            high (int): The upper bound of the distribution.
        """
        super().__init__(float(low - 0.5), float(high + 0.5))

    def discretize(self, n: float) -> int:
        """
        Discretizes a float value to the nearest integer.

        Args:
            n (float): The value to discretize.

        Returns:
            int: The discretized value (rounded to the nearest integer).
        """
        return round(n)


class ChoiceDistribution(Distribution):
    """
    A distribution that returns a choice from a list of possible choices.
    """
    def __init__(self, choices: List[Any]):
        """
        Initializes a ChoiceDistribution with a list of choices.

        Args:
            choices (List[Any]): The list of choices.
        """
        self.choices = choices
        super().__init__(0, len(choices) - 1)

    def discretize(self, n: float) -> Any:
        """
        Discretizes a float value to an element from the choices list.

        Args:
            n (float): The value to discretize (used as an index).

        Returns:
            Any: The element from the choices list at the rounded index.
        """
        return self.choices[round(n)]
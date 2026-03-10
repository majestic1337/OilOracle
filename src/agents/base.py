"""Base agent class."""

from abc import ABC, abstractmethod

from src.core.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute agent logic. Must be implemented by subclasses."""
        pass

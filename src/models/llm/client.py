"""LLM client for inference."""

from src.core.logger import get_logger


class LLMClient:
    """Client for LLM API calls."""

    def __init__(self, model: str = "gpt-4", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self.logger = get_logger("models.llm")

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for given prompt."""
        self.logger.debug("Completing prompt (model=%s)", self.model)
        return ""

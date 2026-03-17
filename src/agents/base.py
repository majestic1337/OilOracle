"""Base agent helpers and shared utilities."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from src.core.logger import get_logger
from src.pipeline.config import PipelineConfig


def get_llm(cfg: PipelineConfig) -> BaseChatModel:
    """Return LangChain chat model based on cfg.llm_provider.

    Supported providers:
      "google"    → ChatGoogleGenerativeAI (default, Google AI Studio)
      "anthropic" → ChatAnthropic
      "lmstudio"  → ChatOpenAI with custom base_url
      "ollama"    → ChatOpenAI with custom base_url
      "openai"    → ChatOpenAI
    """
    if cfg.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.llm_api_key or os.environ["GOOGLE_API_KEY"],
            temperature=0.3,
        )
    if cfg.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key or os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )
    if cfg.llm_provider in ("lmstudio", "ollama"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            api_key="local",
            temperature=0.3,
        )
    if cfg.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key or os.environ.get("OPENAI_API_KEY"),
            temperature=0.3,
        )
    raise ValueError(f"Unknown llm_provider: {cfg.llm_provider}")


class BaseAgent(ABC):
    """Abstract base class for legacy agent patterns."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute agent logic. Must be implemented by subclasses."""
        raise NotImplementedError

"""LLM prompt templates."""

SYSTEM_PROMPT = """You are a financial analysis assistant.
Provide concise, data-driven insights."""


def get_system_prompt(role: str = "analyst") -> str:
    """Return system prompt for given role."""
    return SYSTEM_PROMPT

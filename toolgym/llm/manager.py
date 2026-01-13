"""LLM Manager for creating LLM instances."""

from toolgym.llm.base import BaseLLM, LLMConfig
from toolgym.llm.openrouter import OpenRouterLLM

# Model aliases
MODEL_ALIASES = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "deepseek-v3": "deepseek/deepseek-chat",
    "gemini-2.0": "google/gemini-2.0-flash-exp",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
}


def create_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> BaseLLM:
    """Create an LLM instance via OpenRouter."""
    model_id = MODEL_ALIASES.get(model, model)
    config = LLMConfig(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    return OpenRouterLLM(config)


# Backwards compatible class
class LLMManager:
    MODELS = MODEL_ALIASES
    create = staticmethod(create_llm)
    list_models = staticmethod(lambda: list(MODEL_ALIASES.keys()))

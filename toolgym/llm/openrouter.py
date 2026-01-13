"""OpenRouter LLM implementation - unified access to all models."""

import json
import os
from typing import Any, Optional

from openai import AsyncOpenAI

from toolgym.core.types import Message, Tool, ToolCall
from toolgym.llm.base import BaseLLM, LLMConfig, LLMResponse


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM implementation - access GPT, Claude, etc. via unified API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialize client."""
        if self._client is None:
            api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url or "https://openrouter.ai/api/v1",
                timeout=self.config.timeout,
            )
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        """Generate a response via OpenRouter."""
        kwargs = {
            "model": self.config.model,
            "messages": self._messages_to_api_format(messages),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            kwargs["tools"] = self._tools_to_api_format(tools)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        )

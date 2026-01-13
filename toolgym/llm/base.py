"""Base LLM interface and configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from toolgym.core.types import Message, Tool, ToolCall


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    async def generate_text(self, prompt: str) -> str:
        """Simple text generation from a prompt string. Returns just the text content."""
        from toolgym.core.types import Message, MessageRole
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await self.generate(messages)
        return response.content

    def _messages_to_api_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to API format. Override in subclasses if needed."""
        result = []
        for msg in messages:
            item = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": str(tc.arguments),
                        }
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.name:
                item["name"] = msg.name
            result.append(item)
        return result

    def _tools_to_api_format(self, tools: list[Tool]) -> list[dict]:
        """Convert tools to API format. Override in subclasses if needed."""
        return [tool.to_openai_format() for tool in tools]

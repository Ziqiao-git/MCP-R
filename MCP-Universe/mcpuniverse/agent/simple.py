"""
SimplePromptAgent: ReAct-style control loop but without exposed reasoning.

Differences from ReAct:
- Uses simple_prompt.j2 instead of react_prompt.j2
- The model MUST return valid JSON with EITHER:
    Final answer case:
    {
        "answer": "Final answer text"
    }

    Tool call case:
    {
        "action": {
            "server": "server-name",
            "tool": "tool-name",
            "arguments": {
                "argument-name": "argument-value"
            }
        }
    }

- No "thought", no "reason", no chain-of-thought text in either branch.
- We do NOT require/expect "thought" in parsing.
- We do NOT log "thought" to history or callbacks.

Everything else (looping, HISTORY feeding, tool calls, summarization, callbacks, tracer)
is intentionally kept as close as possible to ReAct.
"""

# pylint: disable=broad-exception-caught
import os
import json
from typing import Optional, Union, Dict, List
from collections import OrderedDict
from dataclasses import dataclass
from mcp.types import TextContent

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.common.logger import get_logger
from mcpuniverse.tracer import Tracer
from mcpuniverse.callbacks.base import (
    send_message,
    send_message_async,
    CallbackMessage,
    MessageType
)
from .base import BaseAgentConfig, BaseAgent
from .utils import build_system_prompt
from .types import AgentResponse

DEFAULT_CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@dataclass
class SimplePromptConfig(BaseAgentConfig):
    """
    Configuration for SimplePromptAgent.

    Attributes:
        system_prompt (str): Template path, should point to simple_prompt.j2.
        context_examples (str): Optional few-shot examples.
        max_iterations (int): Max number of tool/answer loops.
        summarize_tool_response (bool): Whether to summarize tool responses before adding to history.
    """
    system_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "simple_prompt.j2")
    context_examples: str = ""
    max_iterations: int = 5
    summarize_tool_response: bool = False


class SimplePromptAgent(BaseAgent):
    """
    A stripped-down "Act-only" agent:
    - The model either gives an action (tool call) or an answer.
    - No explicit reasoning/thought fields.
    - Multi-turn via HISTORY (same mechanism as ReAct).

    We intentionally keep the control flow and callback structure very close to ReAct
    so you can swap this agent in without touching the rest of the benchmark infra.
    """
    config_class = SimplePromptConfig
    alias = ["simple", "simple_prompt", "direct_act"]

    def __init__(
            self,
            mcp_manager: MCPManager,
            llm: BaseLLM,
            config: Optional[Union[Dict, str]] = None
    ):
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._logger = get_logger(f"{self.__class__.__name__}:{self._name}")
        self._history: List[str] = []

    def _build_prompt(self, question: str):
        """
        Build the prompt for the LLM using simple_prompt.j2 instead of react_prompt.j2.
        We still pass HISTORY, INSTRUCTION, QUESTION, MAX_STEPS, CONTEXT_EXAMPLES, etc.
        """
        params = {
            "INSTRUCTION": self._config.instruction,
            "QUESTION": question,
            "MAX_STEPS": self._config.max_iterations
        }
        if self._config.context_examples:
            params.update({"CONTEXT_EXAMPLES": self._config.context_examples})
        params.update(self._config.template_vars)
        if self._history:
            params.update({"HISTORY": "\n\n".join(self._history)})

        return build_system_prompt(
            system_prompt_template=self._config.system_prompt,
            tool_prompt_template=self._config.tools_prompt,
            tools=self._tools,
            **params
        )

    def _add_history(self, history_type: str, message: str):
        """
        Append a record to conversation HISTORY.
        We still keep the same "Type: message" style as ReAct.
        """
        self._history.append(f"{history_type.title()}: {message}")

    async def _execute(
            self,
            message: Union[str, List[str]],
            output_format: Optional[Union[str, Dict]] = None,
            **kwargs
    ) -> AgentResponse:
        """
        Run the multi-step loop:
        - Ask model for JSON.
        - If "answer": return it.
        - If "action": call tool, stash result in HISTORY, continue.
        - Else: log error and try next iteration.

        This matches ReAct._execute() flow,
        but removes all handling of "thought" fields.
        """
        if isinstance(message, (list, tuple)):
            message = "\n".join(message)
        if output_format is not None:
            message = message + "\n\n" + self._get_output_format_prompt(output_format)

        tracer = kwargs.get("tracer", Tracer())
        callbacks = kwargs.get("callbacks", [])

        for iter_num in range(self._config.max_iterations):
            prompt = self._build_prompt(message)

            response = await self._llm.generate_async(
                messages=[{"role": "user", "content": prompt}],
                tracer=tracer,
                callbacks=callbacks
            )

            try:
                # clean fences etc.
                response = response.strip().strip('`').strip()
                if response.startswith("json"):
                    response = response[4:].strip()

                parsed_response = json.loads(response)

                # mark step in history (keeps parity w/ ReAct's "Step N:")
                self._add_history(
                    history_type=f"Step {iter_num + 1}",
                    message=""
                )

                # CASE 1: final answer
                if "answer" in parsed_response:
                    final_answer = parsed_response["answer"]
                    self._add_history(
                        history_type="answer",
                        message=final_answer
                    )
                    await self._send_callback_message(
                        callbacks=callbacks,
                        iter_num=iter_num,
                        answer=final_answer
                    )
                    return AgentResponse(
                        name=self._name,
                        class_name=self.__class__.__name__,
                        response=final_answer,
                        trace_id=tracer.trace_id
                    )

                # CASE 2: tool call
                if "action" in parsed_response:
                    action = parsed_response["action"]

                    # sanity check shape
                    if not isinstance(action, dict) or "server" not in action or "tool" not in action:
                        # invalid tool block
                        self._add_history(history_type="action", message=str(action))
                        self._add_history(history_type="result", message="Invalid action")
                        await self._send_callback_message(
                            callbacks=callbacks,
                            iter_num=iter_num,
                            action=parsed_response.get("action", ""),
                            result="Invalid action"
                        )
                    else:
                        # log action
                        self._add_history(
                            history_type="action",
                            message=f"Using tool `{action['tool']}` in server `{action['server']}`"
                        )
                        self._add_history(
                            history_type="action input",
                            message=str(action.get("arguments", "none"))
                        )

                        try:
                            tool_result = await self.call_tool(action, tracer=tracer, callbacks=callbacks)
                            tool_content = tool_result.content[0]
                            tool_summary = None

                            if not isinstance(tool_content, TextContent):
                                raise ValueError("Tool output is not a text")

                            if self._config.summarize_tool_response:
                                # summarize tool output so HISTORY doesn't explode
                                context = json.dumps(action, indent=2)
                                tool_summary = await self.summarize_tool_response(
                                    tool_content.text,
                                    context=context,
                                    tracer=tracer
                                )
                                self._add_history(history_type="result", message=tool_summary)
                            else:
                                self._add_history(history_type="result", message=tool_content.text)

                            result_to_log = tool_summary if tool_summary else tool_content.text
                            await self._send_callback_message(
                                callbacks=callbacks,
                                iter_num=iter_num,
                                action=parsed_response.get("action", ""),
                                result=result_to_log
                            )

                        except Exception as e:
                            err_msg = str(e)[:300]
                            self._add_history(history_type="result", message=err_msg)
                            await self._send_callback_message(
                                callbacks=callbacks,
                                iter_num=iter_num,
                                action=parsed_response.get("action", ""),
                                result=err_msg
                            )

                    # after tool call, continue loop to next iteration (let model see updated HISTORY)
                    continue

                # CASE 3: neither "answer" nor "action"
                self._add_history(
                    history_type="error",
                    message="Invalid response format (no 'answer' or 'action')"
                )
                send_message(callbacks, message=CallbackMessage(
                    source=__file__,
                    type=MessageType.LOG,
                    data={
                        "step": iter_num + 1,
                        "error": "Invalid response format (no 'answer' or 'action')"
                    }
                ))

            except json.JSONDecodeError as e:
                # LLM gave non-JSON output
                self._logger.error("Failed to parse response: %s", str(e))
                self._add_history(
                    history_type="error",
                    message="I encountered an error in parsing LLM response. Let me try again."
                )
                send_message(callbacks, message=CallbackMessage(
                    source=__file__,
                    type=MessageType.LOG,
                    data={
                        "step": iter_num + 1,
                        "error": f"Failed to parse response: {str(e)}"
                    }
                ))

            except Exception as e:
                # Catch-all
                self._logger.error("Failed to process response: %s", str(e))
                self._add_history(
                    history_type="error",
                    message="I encountered an unexpected error. Let me try a different approach."
                )
                send_message(callbacks, message=CallbackMessage(
                    source=__file__,
                    type=MessageType.LOG,
                    data={
                        "step": iter_num + 1,
                        "error": f"Failed to process response: {str(e)}"
                    }
                ))

        # ran out of iterations without final answer
        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response="I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations.",
            trace_id=tracer.trace_id
        )

    def get_history(self) -> str:
        """Return conversation history as text."""
        return "\n".join(self._history)

    def clear_history(self):
        """Clear conversation history."""
        self._history = []

    def reset(self):
        """Reset the agent."""
        self.clear_history()

    @staticmethod
    async def _send_callback_message(
            callbacks,
            iter_num: int,
            action: str = "",
            result: str = "",
            answer: str = ""
    ):
        """
        Send callback logs.

        Note: We intentionally DO NOT send "thought" because SimplePromptAgent
        has no reasoning exposure.
        """
        logs = []
        if action:
            logs.append(("action", action))
        if result:
            logs.append(("result", result))
        if answer:
            logs.append(("answer", answer))

        data = OrderedDict({"Iteration": iter_num + 1})
        for tag, value in logs:
            data[tag] = value

        # machine-readable log (sync)
        send_message(callbacks, message=CallbackMessage(
            source=__file__,
            type=MessageType.LOG,
            data=data
        ))

        # pretty console-ish log (async)
        pretty_lines = [
            f"{'=' * 66}\n",
            f"Iteration: {iter_num + 1}\n",
            f"{'-' * 66}\n",
        ]
        for tag, value in logs:
            pretty_lines.append(f"\033[32m{tag.capitalize()}: {value}\n\n\033[0m")

        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=__file__,
                type=MessageType.LOG,
                metadata={
                    "event": "plain_text",
                    "data": "".join(pretty_lines)
                }
            )
        )

#!/usr/bin/env python3
"""
Convert short-horizon trajectory data to ms-swift SFT format.

Uses Hermes function calling format:
- Response format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
- Compatible with Qwen2.5's built-in Hermes tool calling support
"""

import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Simplified ReAct agent instruction
AGENT_INSTRUCTION = """You are a ReAct agent that discovers and uses MCP tools.

Workflow:
1. Use meta-mcp/search_tools to find relevant tools
2. Execute the discovered tools to get data
3. Provide a final answer based on tool results

Always complete both phases: discovery AND execution."""

# Default tools available (meta-mcp search_tools)
DEFAULT_TOOLS_DESCRIPTION = """Server: meta-mcp
Tool: search_tools
Description:
Search for tools by natural language query. Returns tool names, descriptions, and which server they belong to.
Arguments:
- query:
    type: string
    description: Natural language query to search for tools
    required: true
- top_k:
    type: integer
    description: Number of results to return (default: 5)
- min_score:
    type: number
    description: Minimum relevance score threshold (0.0-1.0, default: 0.3)"""


def build_react_prompt(
    question: str,
    history: Optional[str] = None,
    tools_description: str = DEFAULT_TOOLS_DESCRIPTION,
    max_steps: int = 20
) -> str:
    """
    Build prompt matching react_prompt.j2 template format.

    This is what run_react_agent.py actually sends to the LLM.
    """
    tools_prompt = f"""You are able to access to these tools:

### Tools Description ###

{tools_description}

### End of Tools Description ###
"""

    history_section = ""
    if history:
        history_section = f"""Previous steps and results:

{history}"""
    else:
        history_section = "Previous steps and results: EMPTY"

    prompt = f"""You are a ReAct (Reasoning and Acting) agent.
{AGENT_INSTRUCTION}

{tools_prompt}

You need to answer the following question:

Question: {question}

Your goal is to reason about the question and decide on the best course of action to answer it accurately.
You need to choose the appropriate tool based on the question. If no tool is needed, reply directly.
Please use only the tools that are explicitly defined above.

{history_section}

Instructions:
1. Analyze the query, previous reasoning steps, and results.
2. Decide on the next action: use a tool or provide a final answer.
3. Your MUST output the final answer within {max_steps} steps.

If you need to use a tool, first explain your reasoning, then make a tool call using <tool_call> tags:

Your reasoning here...
<tool_call>
{{"name": "tool_name", "arguments": {{"argument_name": "argument_value"}}}}
</tool_call>

If you have enough information to answer the query, provide your final answer directly without using any tool call tags.

Remember:
- Be thorough in your reasoning.
- Use tools when you need more information.
- Always base your reasoning on the actual results from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer when you're confident you have sufficient information.
- Each tool call must be wrapped in <tool_call></tool_call> XML tags.
- **For Pipedream:** If you discover apps with "what_are_you_trying_to_do", you MUST call "select_apps" next to load their tools. After "select_apps" succeeds, the new tools become available for use."""

    return prompt


def extract_server_and_tool(action: str) -> tuple[str, str]:
    """
    Extract server and tool name from action string.

    Input: "Using tool `search_tools` in server `meta-mcp`"
    Output: ("meta-mcp", "search_tools")
    """
    tool_match = re.search(r'Using tool `([^`]+)`', action)
    server_match = re.search(r'in server `([^`]+)`', action)

    tool_name = tool_match.group(1) if tool_match else action
    server_name = server_match.group(1) if server_match else "unknown"

    return server_name, tool_name


def normalize_action_input(action_input: str) -> dict:
    """
    Convert action_input string to dict.
    Handles both Python dict strings (single quotes) and JSON strings (double quotes).
    """
    if isinstance(action_input, dict):
        return action_input

    try:
        # Try JSON first
        return json.loads(action_input)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        # Try Python literal (handles single quotes)
        return ast.literal_eval(action_input)
    except (ValueError, SyntaxError):
        pass

    # Try simple quote replacement
    try:
        fixed = action_input.replace("'", '"')
        return json.loads(fixed)
    except:
        return {}


def format_response_hermes(thought: str, tool: str, arguments: dict) -> str:
    """
    Format response in Hermes function calling format.

    Output format:
    <thought text>
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>
    """
    tool_call = {
        "name": tool,
        "arguments": arguments
    }
    return f"{thought}\n<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"


def format_answer_hermes(thought: str, answer: str) -> str:
    """Format final answer in Hermes format (no tool call, just text)."""
    if thought:
        return f"{thought}\n\n{answer}"
    return answer


def truncate_result(result: str, max_chars: int = 500) -> str:
    """Truncate long results to save context."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n... [truncated, {len(result) - max_chars} chars omitted]"


def parse_reasoning_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse reasoning trace into structured steps.
    Each step has: thought, action, action_input, result
    """
    steps = []
    current_step = {}

    for item in trace:
        item_type = item.get("type", "")
        content = item.get("content", "")

        if item_type.startswith("Step "):
            # New step starts
            if current_step and "action" in current_step:
                steps.append(current_step)
            current_step = {}
        elif item_type == "thought":
            current_step["thought"] = content
        elif item_type == "action":
            current_step["action"] = content
        elif item_type == "action input":
            current_step["action_input"] = content
        elif item_type == "result":
            current_step["result"] = content
        elif item_type == "answer":
            # Final answer - create a special step
            if current_step and "action" in current_step:
                steps.append(current_step)
            current_step = {
                "thought": current_step.get("thought", ""),
                "is_answer": True,
                "answer": content
            }
            steps.append(current_step)
            current_step = {}

    # Don't forget the last step
    if current_step and ("action" in current_step or "is_answer" in current_step):
        steps.append(current_step)

    return steps


def format_history(steps: List[Dict[str, Any]], truncate_chars: int = 500) -> str:
    """
    Format previous steps as history in Hermes-compatible format.

    Format: "Step N:\nThought: ...\nAction: <tool_call>...</tool_call>\nResult: ..."
    """
    history_parts = []

    for i, step in enumerate(steps, 1):
        step_parts = [f"Step {i}:"]

        if step.get("thought"):
            step_parts.append(f"Thought: {step['thought']}")

        if step.get("action"):
            server, tool = extract_server_and_tool(step["action"])
            # Format action in Hermes style (no server)
            action_obj = {
                "name": tool,
                "arguments": normalize_action_input(step.get("action_input", "{}"))
            }
            step_parts.append(f"Action: <tool_call>{json.dumps(action_obj, ensure_ascii=False)}</tool_call>")

        if step.get("result"):
            truncated = truncate_result(step["result"], truncate_chars)
            step_parts.append(f"Result: {truncated}")

        if step.get("is_answer"):
            step_parts.append(f"Answer: {step.get('answer', '')}")

        history_parts.append("\n".join(step_parts))

    return "\n\n".join(history_parts)


def parse_search_tools_result(result: str) -> List[Dict[str, str]]:
    """
    Parse search_tools result to extract discovered tools.

    Input format:
    Found N relevant tools for: '...'

    1. **@server/name** / `tool_name`
       Score: 0.xxx
       Description: ...
       Parameters: param1, param2

    Returns list of dicts with: server, tool, description, parameters
    """
    tools = []

    # Pattern to match each tool entry
    # Format: N. **@server** / `tool_name`
    # Use DOTALL to match multiline descriptions, but stop at "Parameters:"
    pattern = r'(\d+)\.\s+\*\*([^*]+)\*\*\s*/\s*`([^`]+)`\s*\n\s*Score:\s*([\d.]+)\s*\n\s*Description:\s*(.+?)\n\s*Parameters:\s*([^\n]*)'

    matches = re.findall(pattern, result, re.DOTALL)

    for num, server, tool, score, description, parameters in matches:
        # Clean up description (remove extra whitespace and newlines)
        description = ' '.join(description.split())
        tools.append({
            "server": server.strip(),
            "tool": tool.strip(),
            "description": description.strip(),
            "parameters": parameters.strip()
        })

    return tools


def build_tool_description(server: str, tool: str, description: str, parameters: str) -> str:
    """Build a single tool description in the standard format."""
    # Parse parameters string like "param1, param2, param3"
    params = [p.strip() for p in parameters.split(",") if p.strip()]

    if params:
        args_desc = "\n".join([f"- {p}:\n    type: string" for p in params])
    else:
        args_desc = "No arguments specified"

    return f"""Server: {server}
Tool: {tool}
Description:
{description}
Arguments:
{args_desc}"""


def build_tools_description_with_discovered(
    discovered_tools: List[Dict[str, str]]
) -> str:
    """
    Build tools description including discovered tools.

    Args:
        discovered_tools: List of tools discovered via search_tools

    Returns:
        Full tools description string
    """
    # Start with default meta-mcp search_tools
    tools = [DEFAULT_TOOLS_DESCRIPTION]

    # Add discovered tools
    seen = set()
    for tool_info in discovered_tools:
        key = (tool_info["server"], tool_info["tool"])
        if key not in seen:
            seen.add(key)
            tool_desc = build_tool_description(
                tool_info["server"],
                tool_info["tool"],
                tool_info["description"],
                tool_info["parameters"]
            )
            tools.append(tool_desc)

    return "\n\n".join(tools)


def convert_trajectory(traj_data: Dict[str, Any], truncate_chars: int = 2000) -> List[Dict[str, str]]:
    """
    Convert a single trajectory to multiple SFT training samples.

    For each step:
    - Query includes tools discovered in previous steps (via search_tools results)
    - Response is the JSON action for that step

    Returns list of samples, each with 'query' and 'response'.
    """
    user_query = traj_data["metadata"]["query"]
    trace = traj_data["reasoning_trace"]
    max_iterations = traj_data["metadata"].get("max_iterations", 20)

    # Parse into structured steps
    steps = parse_reasoning_trace(trace)

    if not steps:
        return []

    samples = []

    # Track discovered tools as we process steps
    # This simulates the dynamic tool discovery during execution
    discovered_tools: List[Dict[str, str]] = []

    for i, step in enumerate(steps):
        # Skip answer steps for tool call training
        if step.get("is_answer"):
            continue

        # Skip if no action
        if not step.get("action"):
            continue

        # Build tools description with currently discovered tools
        # (tools discovered BEFORE this step, not including this step's result)
        tools_description = build_tools_description_with_discovered(discovered_tools)

        # Build history from previous steps
        history = None
        if i > 0:
            history = format_history(steps[:i], truncate_chars)

        # Build the full prompt (query)
        query = build_react_prompt(
            question=user_query,
            history=history,
            tools_description=tools_description,
            max_steps=max_iterations
        )

        # Build response in Hermes format
        thought = step.get("thought", "")
        server, tool = extract_server_and_tool(step["action"])
        arguments = normalize_action_input(step.get("action_input", "{}"))

        response = format_response_hermes(thought, tool, arguments)

        samples.append({
            "query": query,
            "response": response
        })

        # After processing this step, check if it was a search_tools call
        # If so, parse the result and add discovered tools for future steps
        if tool == "search_tools" and step.get("result"):
            new_tools = parse_search_tools_result(step["result"])
            discovered_tools.extend(new_tools)

    return samples


def convert_all_trajectories(
    input_dir: Path,
    output_file: Path,
    truncate_chars: int = 500
) -> int:
    """
    Convert all trajectory files to SFT format.

    Returns number of samples generated.
    """
    all_samples = []
    traj_files = list(input_dir.rglob("*.json"))

    print(f"Found {len(traj_files)} trajectory files")

    for traj_file in traj_files:
        # Skip .DS_Store and other non-trajectory files
        if traj_file.name.startswith("."):
            continue

        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                traj_data = json.load(f)

            samples = convert_trajectory(traj_data, truncate_chars)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            continue

    # Save as JSONL (ms-swift format)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_samples)} training samples")
    print(f"Saved to {output_file}")

    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories to ms-swift SFT format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/shuang/short-horizon/traj"),
        help="Input directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/shuang/short-horizon/sft_train.jsonl"),
        help="Output JSONL file"
    )
    parser.add_argument(
        "--truncate-chars",
        type=int,
        default=500,
        help="Max characters for result truncation"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Number of example samples to display"
    )

    args = parser.parse_args()

    count = convert_all_trajectories(args.input_dir, args.output, args.truncate_chars)

    if args.show_examples > 0:
        print(f"\n{'='*60}")
        print(f"First {args.show_examples} examples:")
        print(f"{'='*60}")

        with open(args.output, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.show_examples:
                    break
                sample = json.loads(line)
                print(f"\n--- Example {i+1} ---")
                print(f"QUERY (last 1000 chars):\n...{sample['query'][-1000:]}")
                print(f"\nRESPONSE:\n{sample['response']}")
                print("-" * 40)


if __name__ == "__main__":
    main()

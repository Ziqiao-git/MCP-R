#!/usr/bin/env python3
"""
Convert short-horizon trajectory data to ms-swift messages format.
V8 VERSION: Each trajectory is SPLIT into multiple samples (one per step).

This version causes repeated learning:
- traj with 5 steps -> 5 samples
- Each sample includes all previous assistant messages
- ms-swift calculates loss on ALL assistant messages in each sample

Output format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse

# System prompt - the agent instruction
SYSTEM_PROMPT = """You are a ReAct agent that discovers and uses MCP tools.

Workflow:
1. Use meta-mcp/search_tools to find relevant tools
2. Execute the discovered tools to get data
3. Provide a final answer based on tool results

Always complete both phases: discovery AND execution.

Instructions:
1. Analyze the query, previous reasoning steps, and results.
2. Decide on the next action: use a tool or provide a final answer.
3. Your MUST output the final answer within the allowed steps.

If you need to use a tool, first explain your reasoning, then make a tool call using <tool_call> tags:

Your reasoning here...
<tool_call>
{"name": "tool_name", "arguments": {"argument_name": "argument_value"}}
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


def ensure_string(value) -> str:
    """Ensure the value is a string. Convert dict/list to JSON string."""
    if isinstance(value, str):
        return value
    elif isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    elif value is None:
        return ""
    else:
        return str(value)


def truncate_result(result: str, max_chars: int = 500) -> str:
    """Truncate long results to save context."""
    result = ensure_string(result)
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


def format_assistant_tool_call(thought: str, tool: str, arguments: dict) -> str:
    """Format assistant response with tool call."""
    tool_call = {
        "name": tool,
        "arguments": arguments
    }
    return f"{thought}\n<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"


def convert_trajectory_to_messages(
    traj_data: Dict[str, Any],
    truncate_chars: int = 500
) -> List[Dict[str, Any]]:
    """
    Convert a single trajectory to MULTIPLE SFT training samples in messages format.

    V8 VERSION: SPLIT format - each trajectory becomes multiple samples.
    For a trajectory with N steps, we generate N samples:
    - Sample 1: system + user + assistant₁
    - Sample 2: system + user + assistant₁ + tool₁ + assistant₂
    - Sample 3: system + user + assistant₁ + tool₁ + assistant₂ + tool₂ + assistant₃
    - ...

    This causes repeated learning (assistant₁ is learned N times, assistant₂ is learned N-1 times, etc.)

    Returns a list of samples, each with 'messages' list.
    """
    user_query = traj_data["metadata"]["query"]
    trace = traj_data["reasoning_trace"]

    # Parse into structured steps
    steps = parse_reasoning_trace(trace)

    if not steps:
        return []

    samples = []

    # Build base messages (system + user)
    base_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""You are able to access to these tools:

### Tools Description ###

{DEFAULT_TOOLS_DESCRIPTION}

### End of Tools Description ###

Question: {user_query}"""
        }
    ]

    # Accumulate messages as we go through steps
    accumulated_messages = list(base_messages)  # Copy

    for step in steps:
        if step.get("is_answer"):
            # Final answer - ensure it's a string
            answer = ensure_string(step.get("answer", ""))
            thought = ensure_string(step.get("thought", ""))
            if thought:
                assistant_content = f"{thought}\n\n{answer}"
            else:
                assistant_content = answer

            accumulated_messages.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Create sample AFTER adding assistant (final answer has no tool result)
            if len(accumulated_messages) >= 3:
                sample_messages = [dict(m) for m in accumulated_messages]
                samples.append({"messages": sample_messages})

        elif step.get("action"):
            # Tool call step
            thought = ensure_string(step.get("thought", ""))
            _, tool = extract_server_and_tool(step["action"])
            arguments = normalize_action_input(step.get("action_input", "{}"))

            assistant_content = format_assistant_tool_call(thought, tool, arguments)
            accumulated_messages.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Create sample AFTER adding assistant, BEFORE adding tool result
            # This ensures the last message is always assistant
            if len(accumulated_messages) >= 3:
                sample_messages = [dict(m) for m in accumulated_messages]
                samples.append({"messages": sample_messages})

            # Tool result - add to accumulated for next step's context, but NOT in this sample
            if step.get("result"):
                truncated_result = truncate_result(step["result"], truncate_chars)
                accumulated_messages.append({
                    "role": "tool",
                    "content": truncated_result
                })

    return samples


def convert_all_trajectories(
    input_dir: Path,
    output_file: Path,
    truncate_chars: int = 500
) -> Tuple[int, int]:
    """
    Convert all trajectory files to messages format.
    V8 VERSION: Each trajectory becomes MULTIPLE samples (one per step).

    Returns (number of trajectories, number of samples generated).
    """
    all_samples = []
    traj_files = list(input_dir.rglob("*.json"))
    traj_count = 0

    print(f"Found {len(traj_files)} trajectory files")

    for traj_file in traj_files:
        if traj_file.name.startswith("."):
            continue

        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                traj_data = json.load(f)

            samples = convert_trajectory_to_messages(traj_data, truncate_chars)
            if samples:
                traj_count += 1
                all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            continue

    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_samples)} training samples from {traj_count} trajectories")
    print(f"Average samples per trajectory: {len(all_samples) / traj_count:.1f}")
    print(f"Saved to {output_file}")

    return traj_count, len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories to ms-swift messages format (V8 split version)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/shuang/short-horizon/traj"),
        help="Input directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/shuang/short-horizon/sft_train_v8.jsonl"),
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

    traj_count, sample_count = convert_all_trajectories(args.input_dir, args.output, args.truncate_chars)

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
                print(f"Messages count: {len(sample['messages'])}")
                for j, msg in enumerate(sample['messages']):
                    role = msg['role']
                    content = msg['content']
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"\n[{j}] {role.upper()}:")
                    print(content)
                print("-" * 40)


if __name__ == "__main__":
    main()

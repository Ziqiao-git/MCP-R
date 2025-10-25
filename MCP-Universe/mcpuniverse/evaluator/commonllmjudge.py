# mcpuniverse/evaluator/commonllmjudge.py
from __future__ import annotations
import json, os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from mcpuniverse.common.context import Context
from mcpuniverse.evaluator.functions import eval_func, compare_func, FunctionResult

load_dotenv()

# ===================== 基础 utils =====================

def _shorten(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else (s[:n] + "…[trunc]")

def _safe_parse_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse of judge output.
    Returns a dict if we can parse valid JSON, else None.

    Accepts BOTH:
    - legacy style:
        {
          "score": <float 0-1>,
          "explanation": "...",
          "binary": "success"|"failure"
        }

    - new multi-dim style:
        {
          "task_fulfillment": <int 0-10>,
          "grounding": <int 0-10>,
          "tool_choice": <int 0-10>,
          "tool_execution": <int 0-10>,
          "efficiency": <int 0-10>,
          "overall_score": <float 0-1>,
          "explanation": "...",
          "binary": "success"|"failure"
        }

    We do not enforce required keys here; validation is done later.
    """
    if not s:
        return None
    s = s.strip()

    # strip ``` fences and optional "json" tag
    if s.startswith("```"):
        parts = s.split("```")
        candidate = ""
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            candidate = chunk
            break
        s = candidate

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

# ===================== LLM judge prompt =====================

LLM_JUDGE_TEMPLATE = """You are an expert evaluator for multi-step, tool-using ReAct agents.
Your job is to provide a multi-dimensional numeric evaluation of the agent’s reasoning and tool use quality.

You will receive four structured inputs:

1. META:
   - task_id: the benchmark task ID
   - category: the general task domain

2. TASK:
   - question: the original user request the agent tried to solve
   - output_format: the expected schema or structure for the final answer

3. HISTORY:
   The agent’s full reasoning transcript (with "Step i", "Thought:", "Action:", "Result:").
   Treat this as the ground-truth trace of what the agent actually did and saw.
   "Result:" lines are the outputs of tools. Use these as factual evidence.

4. FINAL_ANSWER:
   The final answer produced by the agent.

------------------------------------------------------------
EVALUATION RUBRIC
You must assign integer scores (0–10) for each of the following criteria:

1) Task Fulfillment (0–10)
   - How well does FINAL_ANSWER satisfy the TASK question and required output format?
   - 10 = fully satisfies and correctly structured
   - 5  = partially fulfills OR minor format/schema issues
   - 0  = irrelevant, empty, or clearly wrong task type

2) Grounding (0–10)
   - Are the claims in FINAL_ANSWER supported by evidence in HISTORY (especially tool "Result:" lines)?
   - 10 = every statement in FINAL_ANSWER is justified by HISTORY
   - 5  = mixed / partially justified / some speculation
   - 0  = hallucinated, or contradicts HISTORY

3) Tool Choice (0–10)
   - Did the agent select appropriate tools for the given TASK?
   - 10 = tool(s) chosen are clearly relevant and necessary
   - 5  = somewhat relevant but suboptimal, or missing an obviously helpful tool
   - 0  = tools are clearly irrelevant or no tools were attempted when tools were clearly needed

4) Tool Execution (0–10)
   - Did the tools appear to run successfully and return usable outputs?
   - 10 = all critical tool calls succeeded and were read/used
   - 5  = some tool errors, or agent ignored useful tool output
   - 0  = mostly failures / nonsense / ignored all tool results

5) Efficiency (0–10)
   - Was the reasoning process efficient and purposeful in terms of steps?
   - 10 = minimal but sufficient steps
   - 5  = noticeable redundancy or meandering
   - 0  = chaotic, extremely long, or wildly inefficient for the task

------------------------------------------------------------
OVERALL SCORE
1. Compute overall_score = (task_fulfillment + grounding + tool_choice + tool_execution + efficiency) / 50.
   This must be a float in [0,1].
2. You must still output each individual sub-score.

------------------------------------------------------------
BINARY DECISION
- "binary" is:
    "success"  if overall_score >= {pass_threshold:.2f}
    "failure"  otherwise.

------------------------------------------------------------
OUTPUT FORMAT
Return STRICT JSON with EXACTLY these keys:
{{
  "task_fulfillment": <int 0-10>,
  "grounding": <int 0-10>,
  "tool_choice": <int 0-10>,
  "tool_execution": <int 0-10>,
  "efficiency": <int 0-10>,
  "overall_score": <float 0-1>,
  "explanation": "<short reason referencing HISTORY>",
  "binary": "success" or "failure"
}}

Return ONLY the JSON object. Do NOT include any additional text.

------------------------------------------------------------
DATA:
{payload}
"""

def _call_llm_judge(
    prompt: str,
    *,
    context: Optional[Context] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    prompt_char_limit: int = 15000,
) -> Optional[str]:
    """
    Call the judge model via OpenAI-compatible API.
    """
    ctx = context or Context()
    api_key = ctx.get_env("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = ctx.get_env("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    judge_model = (
        model
        or ctx.get_env("JUDGE_MODEL")
        or os.getenv("JUDGE_MODEL")
        or "openai/gpt-4o-mini"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    safe_prompt = _shorten(prompt, prompt_char_limit)
    tries = 3
    while tries > 0:
        tries -= 1
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict grading judge. "
                            "Always return ONLY the JSON object, "
                            "no prose before or after."
                        ),
                    },
                    {"role": "user", "content": safe_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[commonllmjudge] LLM judge call failed: {e}")
    return None

# ===================== input builder =====================

def _build_inputs(
    llm_response: Any,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    我们约定 benchmark 在调用 evaluator 时，把以下信息塞进 values:
        - "question": 原始用户query (string)
        - "history": ReAct.get_history() 之后的完整文本 (string)
        - "final_answer": AgentResponse.response (string)
        - "output_format": 任务 schema (dict 或 string)
        - "task_id": 可选
        - "category": 可选
        - "correct_answer": gold answer (可选，不强制使用)

    我们不再依赖 tracer，也不再尝试还原 tool_calls。
    """
    question = values.get("question", "")
    history = values.get("history", "")
    final_answer = values.get("final_answer", "")

    task = {
        "task_id": values.get("task_id", "unknown_task"),
        "category": values.get("category", "general"),
        "question": question,
        "output_format": values.get("output_format", {}),
    }

    correct_answer = values.get("correct_answer", "")

    payload = {
        "META": {
            "task_id": task["task_id"],
            "category": task["category"],
            "correct_answer": correct_answer,
        },
        "TASK": {
            "question": task["question"],
            "output_format": task["output_format"],
        },
        "HISTORY": history,
        "FINAL_ANSWER": final_answer,
    }

    return {
        "task": task,
        "history": history,
        "final_answer": final_answer,
        "payload": payload,
        "correct_answer": correct_answer,
    }

# ===================== scoring core =====================

def llm_as_judge_score(
    *,
    task: Dict[str, Any],
    history: str,
    final_answer: str,
    payload: Dict[str, Any],
    judge_model: Optional[str] = None,
    temperature: float = 0.0,
    pass_threshold: float = 0.8,
    context: Optional[Context] = None,
    max_completion_tokens: int = 512,
    prompt_char_limit: int = 15000,
) -> Dict[str, Any]:
    """
    Call judge LLM, parse result, and normalize into our standard contract:
    {
        "score": float in [0,1],
        "binary": "success"|"failure",
        "explanation": str,

        ...plus any extra rubric fields from the judge model...
        (task_fulfillment, grounding, tool_choice, tool_execution, efficiency, overall_score, raw_judge_output)
    }

    If judge output is invalid / unparsable, returns failure w/ score=0.0.
    """

    prompt = LLM_JUDGE_TEMPLATE.format(
        pass_threshold=pass_threshold,
        payload=json.dumps(payload, ensure_ascii=False),
    )

    raw = _call_llm_judge(
        prompt,
        context=context,
        model=judge_model,
        temperature=temperature,
        max_tokens=max_completion_tokens,
        prompt_char_limit=prompt_char_limit,
    )

    parsed = _safe_parse_json(raw)

    if not parsed or not isinstance(parsed, dict):
        return {
            "score": 0.0,
            "binary": "failure",
            "explanation": "Judge model did not return valid JSON.",
            "raw_judge_output": raw,
        }

    # ---------- Case A: legacy style (already had score 0-1) ----------
    if "score" in parsed and "explanation" in parsed:
        try:
            s = float(parsed.get("score", 0.0))
        except Exception:
            s = 0.0
        # clamp
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0

        binary = parsed.get("binary")
        if binary not in ("success", "failure"):
            binary = "success" if s >= pass_threshold else "failure"

        return {
            "score": s,
            "binary": binary,
            "explanation": parsed.get("explanation", ""),
            "raw_judge_output": raw,
            # pass through any extra keys
            **{k: v for k, v in parsed.items() if k not in ("score", "explanation", "binary")},
        }

    # ---------- Case B: new multi-dim style ----------
    task_fulfillment = parsed.get("task_fulfillment")
    grounding = parsed.get("grounding")
    tool_choice = parsed.get("tool_choice")
    tool_execution = parsed.get("tool_execution")
    efficiency = parsed.get("efficiency")
    overall_score = parsed.get("overall_score")

    def _coerce_0_10(x):
        try:
            val = float(x)
        except Exception:
            return None
        if val < 0:
            val = 0
        if val > 10:
            val = 10
        return val

    subs = {
        "task_fulfillment": _coerce_0_10(task_fulfillment),
        "grounding": _coerce_0_10(grounding),
        "tool_choice": _coerce_0_10(tool_choice),
        "tool_execution": _coerce_0_10(tool_execution),
        "efficiency": _coerce_0_10(efficiency),
    }

    def _compute_overall_from_subs(subvals: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [v for v in subvals.values() if v is not None]
        if not vals:
            return None
        # average 0-10 subscores, then /10 to scale to [0,1]
        return (sum(vals) / len(vals)) / 10.0

    try:
        overall = float(overall_score)
    except Exception:
        overall = None

    if overall is None or overall < 0 or overall > 1:
        recomputed = _compute_overall_from_subs(subs)
        overall = recomputed if recomputed is not None else 0.0

    # clamp final overall
    if overall < 0.0:
        overall = 0.0
    if overall > 1.0:
        overall = 1.0

    binary = parsed.get("binary")
    if binary not in ("success", "failure"):
        binary = "success" if overall >= pass_threshold else "failure"

    explanation = parsed.get("explanation", "")

    result = {
        "score": overall,  # <- keeps legacy contract
        "binary": binary,
        "explanation": explanation,
        "raw_judge_output": raw,
        # keep explicit fields for analysis / dashboards
        "task_fulfillment": subs["task_fulfillment"],
        "grounding": subs["grounding"],
        "tool_choice": subs["tool_choice"],
        "tool_execution": subs["tool_execution"],
        "efficiency": subs["efficiency"],
        "overall_score": overall,  # alias
    }

    return result

# ===================== evaluator适配 =====================

def _extract_values_arg(args) -> Dict[str, Any]:
    """
    We expect Evaluator.evaluate to call comparison functions like:
        compare_func(a, extra_values_local, ...kwargs...)

    So inside commonllmjudge_pass / commonllmjudge_score,
    `a` is the agent output (string / AgentResponse-ish),
    and args[0] should be that `extra_values_local` dict we built in Task.evaluate().

    We keep backward compatibility in case someone, somewhere,
    still passes (a, something, values_dict, ...).
    """
    # most common / current case: args[0] is the dict
    if len(args) >= 1 and isinstance(args[0], dict):
        return args[0]

    # fallback: legacy pattern where args[1] held the dict
    if len(args) >= 2 and isinstance(args[1], dict):
        return args[1]

    return {}

@eval_func(name="commonllmjudge.score")
async def commonllmjudge_score(llm_response: Any, *args, **kwargs) -> FunctionResult:
    """
    对单个 AgentResponse 做打分。
    llm_response: 通常是 AgentResponse 或者最终答案字符串
    args[1]: runner 传进来的 values dict (含 question/history/final_answer/...)
    """
    values = _extract_values_arg(args)
    ctx: Context = kwargs.get("context", Context())

    built = _build_inputs(llm_response, values)

    # Step 1. recover final_answer (runner wins, fallback to llm_response)
    final_answer = built["final_answer"]
    if not final_answer:
        if hasattr(llm_response, "response"):
            final_answer = getattr(llm_response, "response")
        else:
            final_answer = str(llm_response)

    # Step 2. sync payload["FINAL_ANSWER"]
    payload = built["payload"].copy()
    payload["FINAL_ANSWER"] = final_answer

    obj = llm_as_judge_score(
        task=built["task"],
        history=built["history"],
        final_answer=final_answer,
        payload=payload,
        judge_model=kwargs.get("judge_model"),
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.8)),
        context=ctx,
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 512)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 15000)),
    )

    return FunctionResult(result=obj)

@compare_func(name="commonllmjudge.pass")
async def commonllmjudge_pass(a: Any, *args, **kwargs) -> (bool, str):
    """
    返回 (ok, reason) ，ok=True表示达到pass_threshold。
    也会顺便把细分分数打印到stderr，方便benchmark日志里看agent表现。
    """
    import sys

    values = _extract_values_arg(args)
    ctx: Context = kwargs.get("context", Context())

    built = _build_inputs(a, values)

    # Step 1. recover final_answer (runner wins, fallback to a)
    final_answer = built["final_answer"]
    if not final_answer:
        if hasattr(a, "response"):
            final_answer = getattr(a, "response")
        else:
            final_answer = str(a)

    # Step 2. sync payload["FINAL_ANSWER"]
    payload = built["payload"].copy()
    payload["FINAL_ANSWER"] = final_answer

    obj = llm_as_judge_score(
        task=built["task"],
        history=built["history"],
        final_answer=final_answer,
        payload=payload,
        judge_model=kwargs.get("judge_model"),
        temperature=float(kwargs.get("temperature", 0.0)),
        pass_threshold=float(kwargs.get("pass_threshold", 0.8)),
        context=ctx,
        max_completion_tokens=int(kwargs.get("max_completion_tokens", 512)),
        prompt_char_limit=int(kwargs.get("prompt_char_limit", 15000)),
    )

    ok = (obj.get("binary") == "success")
    score_val = obj.get("score", None)
    reason = obj.get("explanation", "")

    # print breakdown to stderr so tests / benchmark logs can see the radar
    print(
        "[LLM-JUDGE SCORE] "
        f"overall={score_val} "
        f"binary={obj.get('binary')} "
        f"task_fulfillment={obj.get('task_fulfillment')} "
        f"grounding={obj.get('grounding')} "
        f"tool_choice={obj.get('tool_choice')} "
        f"tool_execution={obj.get('tool_execution')} "
        f"efficiency={obj.get('efficiency')} "
        f"reason={reason}",
        file=sys.stderr
    )

    return ok, reason

@compare_func(name="score>=")
async def score_ge(a: Any, b: Any, *args, **kwargs) -> (bool, str):
    """
    Utility comparator for benchmark YAMLs:
    - left side is typically commonllmjudge.score result, or a dict with "score".
    - right side is threshold, or kwargs["threshold"] fallback.
    """
    if isinstance(a, FunctionResult):
        a = a.result
    if isinstance(a, dict):
        a = a.get("score", 0.0)

    try:
        a_val = float(a)
    except Exception:
        return False, f"invalid score: {a}"

    if isinstance(b, FunctionResult):
        b = b.result
    try:
        b_val = float(b)
    except Exception:
        b_val = float(kwargs.get("threshold", 0.8))

    ok = (a_val >= b_val)
    return (ok, "" if ok else f"score {a_val:.3f} < threshold {b_val:.3f}")


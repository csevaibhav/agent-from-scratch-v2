"""
evals/compare_models.py — Same eval, alag-alag models (model selection by data)
================================================================================

KYUN BANI:
    Step 3 -> 4a-fix mein qwen2.5:7b ki har galti pe prompt badla, aur har baar
    kahin aur kuch toota (regression). Ab sawaal yeh hai: problem hamare DESIGN
    mein hai ya 7B model ki CAPABILITY mein? Jawab andaaze se nahi, experiment se:
        same test set + same prompt + same tools  ->  3 models.
      - Strong model bhi fail -> hamara design galat hai
      - Strong pass, 7B fail  -> model ki capability ki limit hai

TRAJECTORY CHECKS (pehle se mazboot, kyunki Step 4a-fix mein bug miss hua tha):
    Math cases pe:
      1. required_tools : get_current_time + calculate DONO chale?
      2. order          : time PEHLE fetch hua, calculate BAAD mein?
      3. sequential     : ek step mein EK hi calculate? (ek saath kai = agle ka
                          input guess kiya hoga, jaise "776" wala bug)
    No-tool cases pe:
      4. koi tool nahi, aur 1 step

METRICS (har model):
    answer pass, trajectory pass, no-tool pass, avg steps, avg tokens, avg latency

KAISE CHALAYEIN:
    python -m evals.compare_models                       # teeno models
    python -m evals.compare_models ollama:qwen3:8b       # sirf ek model

INTERVIEW LINE:
    "Model gut feel se nahi chuna. Apne eval suite pe 3 models compare kiye:
    correctness, trajectory, no-tool discipline, latency aur tokens."
"""

import re
import sys
import time
from datetime import datetime

from llm.client import LLMClient
from raw_agent.agent import SYSTEM_PROMPT, run_agent

RUNS = 3   # har case kitni baar

DEFAULT_MODELS = [
    "ollama:qwen2.5:7b",
    "ollama:qwen3:8b",
    "anthropic:claude-haiku-5-5",
]

# Math cases: dependent hain. Pehle time chahiye, phir 2 operations ek ke baad ek.
MATH_CASES = [
    {"q": "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it.",
     "expected": lambda now: now.hour * 1000 + 7006751},
    {"q": "Take today's day of the month, multiply it by 250, and then subtract 99.",
     "expected": lambda now: now.day * 250 - 99},
    {"q": "Take the current year, subtract 1947 from it, and multiply the result by 12.",
     "expected": lambda now: (now.year - 1947) * 12},
]

# No-tool cases: kuch maanga hi nahi gaya.
NO_TOOL_CASES = ["My name is Vaibhav.", "I was born in 2000.", "Thanks, that was helpful!"]


# ------------------------------------------------------------------
# CHECKERS
# ------------------------------------------------------------------
def answer_is_correct(answer: str, expected: int) -> bool:
    numbers = re.findall(r"-?\d+", (answer or "").replace(",", ""))
    return str(expected) in numbers


def tool_steps(messages: list) -> list[list[str]]:
    """Har assistant step mein kaunse tools maange gaye: [["get_current_time"], ["calculate"], ...]"""
    return [[tc["function"]["name"] for tc in m.get("tool_calls", [])]
            for m in messages if m["role"] == "assistant" and m.get("tool_calls")]


def trajectory_check(messages: list) -> tuple[bool, str]:
    """Math case ki trajectory. Return: (pass?, reason)."""
    steps = tool_steps(messages)
    flat = [t for s in steps for t in s]

    # 1. Zaroori tools
    if "get_current_time" not in flat:
        return False, "get_current_time NAHI chala (stale knowledge ka risk)"
    if "calculate" not in flat:
        return False, "calculate NAHI chala (khud math kiya)"

    # 2. Order: time wala step, pehle calculate wale step se PEHLE
    time_step = next(i for i, s in enumerate(steps) if "get_current_time" in s)
    calc_step = next(i for i, s in enumerate(steps) if "calculate" in s)
    if time_step >= calc_step:
        return False, "time aur calculate ek saath / galat order (input guess hua)"

    # 3. Sequential: ek step mein ek se zyada calculate = dependent input guess
    if any(s.count("calculate") > 1 for s in steps):
        return False, "ek step mein kai calculate (dependent input fabricate)"

    return True, "ok"


# ------------------------------------------------------------------
# EK MODEL KA EVAL
# ------------------------------------------------------------------
def eval_model(spec: str) -> dict:
    provider, model = spec.split(":", 1)   # "ollama:qwen3:8b" -> ("ollama", "qwen3:8b")
    client = LLMClient(provider=provider, model=model)
    print("=" * 78)
    print(f"MODEL: {spec}")

    stats = {"answer": 0, "traj": 0, "math_total": 0, "notool": 0, "notool_total": 0,
             "steps": [], "tokens": [], "latency": []}

    def run(q):
        t0 = time.perf_counter()
        r = run_agent(q, system_prompt=SYSTEM_PROMPT, verbose=False, llm_client=client)
        stats["latency"].append(time.perf_counter() - t0)
        stats["steps"].append(r["steps"])
        stats["tokens"].append(r["tokens"]["input"] + r["tokens"]["output"])
        return r

    # ---- Math cases ----
    for case in MATH_CASES:
        for i in range(1, RUNS + 1):
            expected = case["expected"](datetime.now())
            r = run(case["q"])
            ok_ans = answer_is_correct(r["answer"], expected)
            ok_traj, reason = trajectory_check(r["messages"])
            stats["math_total"] += 1
            stats["answer"] += ok_ans
            stats["traj"] += ok_traj
            print(f"  [math] {case['q'][:42]:42}.. run {i}: ans={'PASS' if ok_ans else 'FAIL'} "
                  f"traj={'PASS' if ok_traj else 'FAIL'} steps={r['steps']}"
                  + ("" if ok_traj else f"  <- {reason}"))
            if not ok_ans:
                print(f"         expected={expected} | got: {(r['answer'] or '')[:100]!r}")
                print(f"         tool steps: {tool_steps(r['messages'])}")

    # ---- No-tool cases ----
    for msg in NO_TOOL_CASES:
        for i in range(1, RUNS + 1):
            r = run(msg)
            ok = not tool_steps(r["messages"]) and r["steps"] == 1
            stats["notool_total"] += 1
            stats["notool"] += ok
            print(f"  [no-tool] {msg!r:30} run {i}: {'PASS' if ok else 'FAIL'}"
                  + ("" if ok else f"  tools={tool_steps(r['messages'])}"))

    return stats


def avg(xs):
    return sum(xs) / len(xs) if xs else 0


if __name__ == "__main__":
    models = sys.argv[1:] or DEFAULT_MODELS
    results = {}
    for spec in models:
        try:
            results[spec] = eval_model(spec)
        except Exception as e:   # ek model fail ho (jaise API key galat), toh baaki chalein
            print(f"  !! {spec} chal nahi paya: {e}")

    # ---- Final comparison table ----
    print("\n" + "=" * 78)
    print(f"{'MODEL':30} {'ANSWER':>8} {'TRAJ':>8} {'NO-TOOL':>8} {'STEPS':>6} {'TOKENS':>7} {'SEC':>6}")
    for spec, s in results.items():
        print(f"{spec:30} {s['answer']:>3}/{s['math_total']:<4} {s['traj']:>3}/{s['math_total']:<4} "
              f"{s['notool']:>3}/{s['notool_total']:<4} {avg(s['steps']):>6.1f} "
              f"{avg(s['tokens']):>7.0f} {avg(s['latency']):>6.1f}")
"""
raw_agent/step3b_reliability.py — Phase 1, Step 3b/3c: reliability eval
=======================================================================

KYUN BANI:
    Step 3 mein model ne calculator skip karke galat math kiya.
    Ek sawaal ko N baar chala ke PASS RATE measure karte hain, kyunki
    LLM non-deterministic hai (temperature 0 pe bhi).

STEP 3c UPDATE (Step 3b ki galtiyon se seekh):
    1. TEST LEAKAGE hataya: naye prompt ka example neutral hai, aur test
       ALAG-ALAG sawaalon pe hai jo prompt mein kahin nahi hain (generalization).
    2. FAIL hone pe POORA jawab print hota hai, taaki pata chale galti model ki
       thi ya hamare checker ki (eval code bhi buggy ho sakta hai).

DO CHECKS (har run pe):
    1. Answer check     : final jawab mein sahi number hai?  (ground truth CODE se)
    2. Trajectory check : calculator SACH MEIN call hua?     (process sahi tha?)

KAISE CHALAYEIN:
    python -m raw_agent.step3b_reliability

INTERVIEW LINE:
    "Prompt change ko pass rate se measure kiya, answer aur trajectory dono pe,
    aur test leakage se bachne ke liye unseen sawaalon pe test kiya."
"""

import re
from datetime import datetime

from raw_agent.agent import SYSTEM_PROMPT, run_agent

RUNS_PER_QUESTION = 3

# Step 3 wala purana (soft) prompt, sirf comparison ke liye yahan rakha hai.
OLD_PROMPT = (
    "You are a helpful assistant that solves tasks step by step using tools.\n"
    "- Use the calculator for ANY arithmetic; never calculate in your head.\n"
    "- Use get_current_time whenever the task depends on the current date or time.\n"
    "- If a step depends on the result of an earlier tool, wait for that result first.\n"
    "- When you have everything you need, give a short final answer in plain text "
    "(no LaTeX, no markdown)."
)

# Test set: har sawaal ke saath uska sahi jawab nikalne ka function.
# Sahi jawab CODE se nikalte hain, model se nahi. Yahi "ground truth" hai.
# Teeno sawaal DEPENDENT hain: pehle time/date chahiye, phir calculation.
TEST_CASES = [
    {
        "question": "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it.",
        "expected": lambda now: now.hour * 1000 + 7006751,
    },
    {
        "question": "Take today's day of the month, multiply it by 250, and then subtract 99.",
        "expected": lambda now: now.day * 250 - 99,
    },
    {
        "question": "Take the current year, subtract 1947 from it, and multiply the result by 12.",
        "expected": lambda now: (now.year - 1947) * 12,
    },
]


def answer_is_correct(answer: str, expected: int) -> bool:
    """Jawab ke saare numbers nikalo (commas hata ke, minus sign samet) aur sahi number dhoondho."""
    numbers = re.findall(r"-?\d+", (answer or "").replace(",", ""))
    return str(expected) in numbers


def calculator_was_used(messages: list) -> bool:
    """History mein dekho: kya kisi assistant message ne calculator tool maanga tha?"""
    for m in messages:
        for tc in m.get("tool_calls", []):
            if tc["function"]["name"] == "calculator":
                return True
    return False


def evaluate(prompt_name: str, prompt: str):
    print("=" * 70)
    print(f"PROMPT: {prompt_name}")
    total = answer_pass = traj_pass = 0

    for case in TEST_CASES:
        print(f"\n  Q: {case['question']}")
        for i in range(1, RUNS_PER_QUESTION + 1):
            expected = case["expected"](datetime.now())
            result = run_agent(case["question"], system_prompt=prompt, verbose=False)
            ok_answer = answer_is_correct(result["answer"], expected)
            ok_traj = calculator_was_used(result["messages"])
            total += 1
            answer_pass += ok_answer
            traj_pass += ok_traj

            print(f"    run {i}: answer={'PASS' if ok_answer else 'FAIL'} | "
                  f"calculator_used={'YES' if ok_traj else 'NO '} | steps={result['steps']} | "
                  f"expected={expected}")
            if not ok_answer:
                # FAIL pe poora jawab: model ki galti thi ya checker ki, yeh dekhna hai.
                print(f"      FULL ANSWER: {result['answer']!r}")

    print(f"\n  --> Answer pass rate     : {answer_pass}/{total}")
    print(f"  --> Trajectory pass rate : {traj_pass}/{total}")


if __name__ == "__main__":
    evaluate("OLD (soft rules)", OLD_PROMPT)
    evaluate("NEW (strict, neutral example)", SYSTEM_PROMPT)
"""
raw_agent/step3b_reliability.py — Phase 1, Step 3b: pehla mini-eval
===================================================================

KYUN BANI:
    Step 3 mein model ne calculator skip karke galat math kiya.
    "Prompt badal ke ek baar dekh lo" kaafi nahi hai. LLM non-deterministic hai,
    ek baar sahi aana luck ho sakta hai. Isliye MEASURE karte hain:
    ek sawaal N baar chalao, code se check karo, PASS RATE nikalo.

DO CHECKS (har run pe):
    1. Answer check     : final jawab mein sahi number hai?
    2. Trajectory check : calculator SACH MEIN call hua? (process sahi tha?)
       Sirf answer check karte toh lucky guess bhi pass ho jaata.

KYA COMPARE HO RAHA HAI:
    OLD prompt (Step 3 wala) vs STRICT prompt (zyada explicit rules)

KAISE CHALAYEIN:
    python -m raw_agent.step3b_reliability

INTERVIEW LINE:
    "Maine prompt change ko gut-feel se nahi, pass rate se measure kiya.
    Answer correctness aur trajectory (sahi tool use hua?) dono check kiye."
"""

import re
from datetime import datetime

from raw_agent.agent import SYSTEM_PROMPT, run_agent

QUESTION = "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it."
RUNS = 5

# STRICT prompt: rules ko explicit, numbered, aur example ke saath likha.
STRICT_PROMPT = (
    "You are a helpful assistant that solves tasks step by step using tools.\n"
    "STRICT RULES:\n"
    "1. You MUST NOT do any arithmetic yourself, not even simple addition or multiplication. "
    "Every calculation MUST go through the calculator tool.\n"
    "2. Work in order: first get any data you need (for example the current time), "
    "then call the calculator with ONE full expression using that data, "
    "for example '4 * 1000 + 7006751'.\n"
    "3. Only give the final answer AFTER the calculator has returned its result. "
    "Use exactly the number the calculator returned.\n"
    "4. Final answer in plain text (no LaTeX, no markdown)."
)


def expected_answer() -> int:
    """Sahi jawab hum CODE se nikalte hain, model se nahi. Yahi eval ka 'ground truth' hai."""
    return datetime.now().hour * 1000 + 7006751


def answer_is_correct(answer: str, expected: int) -> bool:
    """Jawab ke saare numbers nikalo (commas hata ke) aur dekho ki sahi number unme hai ya nahi."""
    numbers = re.findall(r"\d+", (answer or "").replace(",", ""))
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
    print(f"PROMPT: {prompt_name}  ({RUNS} runs)")
    answer_pass, traj_pass = 0, 0

    for i in range(1, RUNS + 1):
        expected = expected_answer()
        result = run_agent(QUESTION, system_prompt=prompt, verbose=False)
        ok_answer = answer_is_correct(result["answer"], expected)
        ok_traj = calculator_was_used(result["messages"])
        answer_pass += ok_answer
        traj_pass += ok_traj

        print(f"  run {i}: answer={'PASS' if ok_answer else 'FAIL'} | "
              f"calculator_used={'YES' if ok_traj else 'NO '} | steps={result['steps']} | "
              f"expected={expected} | got: {result['answer'][:80]!r}")

    print(f"  --> Answer pass rate     : {answer_pass}/{RUNS}")
    print(f"  --> Trajectory pass rate : {traj_pass}/{RUNS}")


if __name__ == "__main__":
    evaluate("OLD (Step 3)", SYSTEM_PROMPT)
    evaluate("STRICT", STRICT_PROMPT)
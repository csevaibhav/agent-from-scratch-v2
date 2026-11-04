"""
raw_agent/agent.py — Phase 1, Step 3: ReAct agent loop (bina framework ke)
==========================================================================

KYUN BANI:
    Step 2 ka ek round trip sirf ek tool tak kaam karta tha. Asli sawaalon mein
    model ko ek ke baad ek kai tools chahiye hote hain, aur agla tool pichhle ke
    result pe depend karta hai. Isliye round trip ko LOOP bana diya. Yahi agent hai.

ReAct = REASON -> ACT -> OBSERVE (repeat):
    Reason  : model history padh ke sochta hai
    Act     : tool maangta hai (tool_calls)
    Observe : hamara code tool chalata hai, result history mein jaata hai
    ...jab tak model bina tool ke final jawab na de de.

STOP CONDITIONS (interview mein zaroor poochte hain):
    1. Model ne tool nahi maanga  -> final answer
    2. max_iterations cross       -> infinite loop / runaway cost se bachao

KAISE CHALAYEIN:
    python -m raw_agent.agent

LANGCHAIN / LANGGRAPH MAPPING:
    - Yeh poora run_agent() = LangChain ka AgentExecutor / create_agent
    - LangGraph mein: "agent" node -> conditional edge (tool_calls? -> "tools" node : END)
      -> "tools" node se wapas "agent" node. Wahi loop, bas graph ki shakal mein.
"""

from llm.client import LLMClient
from raw_agent.tools import TOOL_SCHEMAS, execute_tool

# System prompt: agent ka "job description".
# Step 3 mein seekha: soft instruction ("never calculate in your head") ko
# qwen2.5:7b ne ignore kiya. Isliye ab STRICT, numbered rules.
# DHYAN: example jaan-boojh ke NEUTRAL hai ('12 * 30 + 7'), test sawaalon jaisa nahi.
# Warna test leakage hota hai (Step 3b ki galti).
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves tasks step by step using tools.\n"
    "STRICT RULES:\n"
    "1. You MUST NOT do any arithmetic yourself, not even simple addition or multiplication. "
    "Every calculation MUST go through the calculator tool.\n"
    "2. Work in order: first get any data you need (for example the current date or time), "
    "then call the calculator with ONE full expression using that data, "
    "for example '12 * 30 + 7'.\n"
    "3. Only give the final answer AFTER the calculator has returned its result. "
    "Use exactly the number the calculator returned.\n"
    "4. Final answer in plain text (no LaTeX, no markdown)."
)

llm = LLMClient()


def run_agent(question: str, system_prompt: str = SYSTEM_PROMPT,
              max_iterations: int = 6, verbose: bool = True) -> dict:
    """
    Ek sawaal leke ReAct loop chalata hai.

    RETURN (dict), taaki evals/traces result ko CODE se check kar sakein:
        answer   : final jawab (str)
        messages : poori history (kaunsa tool kab call hua, sab isme hai)
        steps    : kitni LLM calls lagi
        tokens   : {"input", "output"}

    system_prompt  : parameter isliye, taaki alag-alag prompts compare kar sakein (Step 3b)
    max_iterations : safety limit
    verbose        : har step print karna (Phase 2 mein yeh tracer banega)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    total_in, total_out = 0, 0

    for step in range(1, max_iterations + 1):
        # ---------- REASON ----------
        resp = llm.chat(messages, tools=TOOL_SCHEMAS)
        total_in += resp.usage.get("input_tokens", 0)
        total_out += resp.usage.get("output_tokens", 0)
        messages.append(llm.assistant_message(resp))

        # ---------- STOP CONDITION 1: koi tool nahi = final answer ----------
        if not resp.tool_calls:
            if verbose:
                print(f"[step {step}] FINAL ANSWER")
                print(f"[tokens] input={total_in} output={total_out}")
            return {"answer": resp.content, "messages": messages, "steps": step,
                    "tokens": {"input": total_in, "output": total_out}}

        # ---------- ACT + OBSERVE ----------
        for tc in resp.tool_calls:
            result = execute_tool(tc["name"], tc["arguments"])
            messages.append(llm.tool_result_message(tc["id"], result))
            if verbose:
                print(f"[step {step}] ACT  : {tc['name']}({tc['arguments']})")
                print(f"[step {step}] OBS  : {result}")

    # ---------- STOP CONDITION 2: limit cross ----------
    return {"answer": f"Stopped: {max_iterations} steps mein jawab nahi mila.",
            "messages": messages, "steps": max_iterations,
            "tokens": {"input": total_in, "output": total_out}}


if __name__ == "__main__":
    tests = [
        "What is 1234 * 5678 + 99? Is it even or odd?",
        "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it.",
        "In one line, what is the capital of Japan?",
    ]
    for q in tests:
        print("=" * 70)
        print("USER:", q)
        result = run_agent(q)
        print("ANSWER:", result["answer"])   # ab dict hai, isliye ["answer"]
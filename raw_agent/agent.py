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
# Yahan behaviour aur output format dono control karte hain.
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves tasks step by step using tools.\n"
    "- Use the calculator for ANY arithmetic; never calculate in your head.\n"
    "- Use get_current_time whenever the task depends on the current date or time.\n"
    "- If a step depends on the result of an earlier tool, wait for that result first.\n"
    "- When you have everything you need, give a short final answer in plain text "
    "(no LaTeX, no markdown)."
)

llm = LLMClient()


def run_agent(question: str, max_iterations: int = 6, verbose: bool = True) -> str:
    """
    Ek sawaal leke ReAct loop chalata hai aur final jawab lautata hai.
    max_iterations : kitni baar tak LLM call karne denge (safety limit)
    verbose        : har step print karna hai ya nahi (abhi debugging; Phase 2 mein yeh tracer banega)
    """
    # Short-term memory = yeh messages list. Har step isme judta hai.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    total_in, total_out = 0, 0   # poore run ke tokens (cost ka andaaza)

    for step in range(1, max_iterations + 1):
        # ---------- REASON: model history padh ke decide karta hai ----------
        resp = llm.chat(messages, tools=TOOL_SCHEMAS)
        total_in += resp.usage.get("input_tokens", 0)
        total_out += resp.usage.get("output_tokens", 0)

        # Model ka har response history mein jodo, final wala bhi.
        # (Step 2 mein yeh missing tha, isliye follow-up sawaal toot jaata.)
        messages.append(llm.assistant_message(resp))

        # ---------- STOP CONDITION 1: koi tool nahi maanga = kaam khatam ----------
        if not resp.tool_calls:
            if verbose:
                print(f"[step {step}] FINAL ANSWER")
                print(f"[tokens] input={total_in} output={total_out} total={total_in + total_out}")
            return resp.content

        # ---------- ACT + OBSERVE: har maanga gaya tool chalao ----------
        for tc in resp.tool_calls:
            result = execute_tool(tc["name"], tc["arguments"])                 # ACT (hamara code)
            messages.append(llm.tool_result_message(tc["id"], result))         # OBSERVE (result history mein)
            if verbose:
                print(f"[step {step}] ACT  : {tc['name']}({tc['arguments']})")
                print(f"[step {step}] OBS  : {result}")

    # ---------- STOP CONDITION 2: limit cross ----------
    if verbose:
        print(f"[tokens] input={total_in} output={total_out} total={total_in + total_out}")
    return f"Stopped: {max_iterations} steps mein jawab nahi mila."


# Sirf tab chalta hai jab file seedha run ho (import hone pe nahi).
if __name__ == "__main__":
    tests = [
        # 1 tool: simple
        "What is 1234 * 5678 + 99? Is it even or odd?",
        # 2 tools, DEPENDENT: pehle time chahiye, tabhi calculation ho sakti hai
        "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it.",
        # 0 tools
        "In one line, what is the capital of Japan?",
    ]
    for q in tests:
        print("=" * 70)
        print("USER:", q)
        answer = run_agent(q)
        print("ANSWER:", answer)
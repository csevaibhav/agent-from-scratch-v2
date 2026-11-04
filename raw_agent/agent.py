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
# Step 4a fix: purana prompt "solves tasks step by step using tools" se shuru hota tha,
# isliye model har message (jaise "My name is Vaibhav.") ko TASK samajh ke tools
# chalane laga, 2407 tokens kharch kiye, aur ek adhoora jawab diya.
# Ab PEHLA decision: "user ne kuch maanga bhi hai ya nahi?"
# Leakage se bachav: yahan concrete example NAHI, sirf categories. Test statements kahin aur hain.
SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools.\n"
    "FIRST decide whether the user's latest message actually asks for something.\n"
    "- If it only shares information (such as a fact about themselves), is a greeting, "
    "or says thanks, reply briefly and naturally. Do NOT call any tool and do NOT invent a task.\n"
    "- Use tools ONLY when the user asks a question or gives a task that needs them.\n"
    "STRICT RULES when a task needs arithmetic or the current date/time:\n"
    "1. You MUST NOT do any arithmetic yourself, not even simple addition or subtraction. "
    "Every single operation MUST go through the calculate tool.\n"
    "2. First get any data you need (for example the current date or time).\n"
    "3. Then do the operations ONE AT A TIME, in the order the task describes, "
    "using the previous result as the next input.\n"
    "4. Give the final answer only AFTER the last calculate result, "
    "in plain text (no LaTeX, no markdown)."
)

llm = LLMClient()


class Agent:
    """
    ReAct agent jo SHORT-TERM MEMORY rakhta hai.

    KYUN CLASS:
        Function har call pe naya messages list banata tha, toh pichhla sawaal
        bhool jaata tha. Class mein self.messages har ask() ke beech zinda rehta hai.
        Isliye follow-up sawaal ("mera naam kya hai?") ka jawab de pata hai.

    SHORT-TERM MEMORY ASAL MEIN:
        Bas self.messages list. Har LLM call mein POORI list jaati hai.
        Keemat: har turn ke saath tokens badhte hain (context window ki limit aayegi).

    LANGGRAPH MAPPING:
        self.messages = graph ki State; ek Agent object = ek thread_id;
        reset() = naya thread_id. Checkpointer yahi state save/load karta hai.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT,
                 max_iterations: int = 6, verbose: bool = True):
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Memory saaf karo: sirf system prompt bachta hai (naya thread)."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def ask(self, question: str) -> dict:
        """
        Ek user sawaal leke ReAct loop chalata hai. History self.messages mein BANI RAHTI hai.
        RETURN: {"answer", "messages", "steps", "tokens"}, jaise pehle (evals ke liye).
        """
        self.messages.append({"role": "user", "content": question})
        total_in, total_out = 0, 0

        for step in range(1, self.max_iterations + 1):
            # ---------- REASON: poori history (pichhle turns samet) jaati hai ----------
            resp = llm.chat(self.messages, tools=TOOL_SCHEMAS)
            total_in += resp.usage.get("input_tokens", 0)
            total_out += resp.usage.get("output_tokens", 0)
            self.messages.append(llm.assistant_message(resp))

            # ---------- STOP CONDITION 1: koi tool nahi = final answer ----------
            if not resp.tool_calls:
                if self.verbose:
                    print(f"[step {step}] FINAL ANSWER | tokens in={total_in} out={total_out}")
                return {"answer": resp.content, "messages": self.messages, "steps": step,
                        "tokens": {"input": total_in, "output": total_out}}

            # ---------- ACT + OBSERVE ----------
            for tc in resp.tool_calls:
                result = execute_tool(tc["name"], tc["arguments"])
                self.messages.append(llm.tool_result_message(tc["id"], result))
                if self.verbose:
                    print(f"[step {step}] ACT : {tc['name']}({tc['arguments']})")
                    print(f"[step {step}] OBS : {result}")

        # ---------- STOP CONDITION 2: limit cross ----------
        return {"answer": f"Stopped: {self.max_iterations} steps mein jawab nahi mila.",
                "messages": self.messages, "steps": self.max_iterations,
                "tokens": {"input": total_in, "output": total_out}}


def run_agent(question: str, system_prompt: str = SYSTEM_PROMPT,
              max_iterations: int = 6, verbose: bool = True) -> dict:
    """
    ONE-SHOT helper: naya Agent, ek sawaal, khatam. Koi memory nahi.
    Kyun rakha: evals (step3b_reliability) har run FRESH chahte hain. Pichhle run
    ki history agle run ko influence na kare, warna test fair nahi rahega.
    """
    return Agent(system_prompt, max_iterations, verbose).ask(question)


if __name__ == "__main__":
    tests = [
        "What is 1234 * 5678 + 99? Is it even or odd?",
        "Find the current hour of the day, multiply it by 1000, and then add 7006751 to it.",
        "In one line, what is the capital of Japan?",
    ]
    for q in tests:
        print("=" * 70)
        print("USER:", q)
        print("ANSWER:", run_agent(q)["answer"])
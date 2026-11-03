"""
raw_agent/step2_round_trip.py — Phase 1, Step 2: ek poora tool-calling round trip
================================================================================

KYUN BANI:
    Step 1 mein tool chala, par result model tak wapas nahi gaya.
    Yahan poora cycle banate hain:
        LLM call #1 (tool decide) -> hum tool chalate hain -> LLM call #2 (final jawab)

SABSE IMPORTANT CONCEPT:
    LLM STATELESS hai. Use pichli call yaad nahi rehti. Isliye doosri call mein
    POORI history bhejte hain:
        user message  ->  assistant ka tool_call message  ->  tool result message

KAISE CHALAYEIN:
    python -m raw_agent.step2_round_trip

INTERVIEW LINE:
    "Tool calling = kam se kam 2 LLM calls. Pehli tool decide karti hai, hum execute
    karte hain, doosri result padh ke user ko jawab deti hai."
"""

import json

from llm.client import LLMClient
from raw_agent.tools import TOOL_SCHEMAS, execute_tool

llm = LLMClient()

question = "What is 1234 * 5678 + 99? Also tell me whether the result is even or odd."

# Conversation history. Har message is list mein judta jayega.
# Yahi list agent ki SHORT-TERM MEMORY hai (Step 4 mein detail mein).
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": question},
]

# ---------------- LLM CALL #1: model tool decide karega ----------------
resp1 = llm.chat(messages, tools=TOOL_SCHEMAS)
print("CALL #1 tool_calls:", resp1.tool_calls)

if not resp1.tool_calls:
    print("Model ne tool nahi maanga, seedha jawab:", resp1.content)
else:
    # (a) Model ka apna tool-call message history mein daalo.
    #     OpenAI format mein yeh exact shape chahiye. Isliye apne simple dict
    #     ko wapas OpenAI shape mein badal rahe hain.
    #     NOTE: yeh conversion provider-specific hai (Anthropic ka format alag hai).
    #     Aage ise llm/client.py mein le jayenge, taaki agent ko format na pata ho.
    messages.append({
        "role": "assistant",
        "content": resp1.content or "",
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in resp1.tool_calls
        ],
    })

    # (b) Har tool chalao, aur uska result "tool" role ke message mein daalo.
    for tc in resp1.tool_calls:
        result = execute_tool(tc["name"], tc["arguments"])
        print(f"TOOL RUN: {tc['name']} -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],   # kis call ka result hai, yeh id batati hai
            "content": result,          # result hamesha string
        })

    # ---------------- LLM CALL #2: model result padh ke final jawab dega ----------------
    resp2 = llm.chat(messages, tools=TOOL_SCHEMAS)
    print("CALL #2 tool_calls:", resp2.tool_calls)   # ideally [] (kaam ho gaya)
    print("FINAL ANSWER:", resp2.content)

    # Tokens dekho: call #2 mein history badi hai, isliye input tokens zyada honge.
    print("TOKENS call#1:", resp1.usage, "| call#2:", resp2.usage)

# Poori history print karo, taaki dikhe model ko asal mein kya-kya bheja gaya.
print("\n----- FULL MESSAGE HISTORY -----")
print(json.dumps(messages, indent=2, ensure_ascii=False))
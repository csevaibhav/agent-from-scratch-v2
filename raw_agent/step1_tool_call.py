"""
raw_agent/step1_tool_call.py — Phase 1, Step 1: raw tool calling dekhna
=======================================================================

KYUN BANI:
    Apni aankhon se dekhna ki tool calling mein model ASAL mein kya lautata hai.
    Model function nahi chalata, sirf JSON order deta hai.

KYA KARTI HAI:
    3 sawaal bhejti hai, tools ke schema ke saath:
      1. Math sawaal    -> model ko calculator chunna chahiye
      2. Time sawaal    -> model ko get_current_time chunna chahiye
      3. General sawaal -> model ko KOI tool nahi chunna chahiye, seedha jawab
    Phir model ka raw tool_call print karti hai, aur HAMARA code tool chalata hai.

KAISE CHALAYEIN (project root se):
    python -m raw_agent.step1_tool_call
    ("-m" isliye, taaki Python root folder se chale aur "llm" package mil jaaye)

ABHI KYA NAHI HO RAHA:
    Tool ka result wapas model ko nahi bhej rahe. Woh Step 2 mein hoga
    (yaani ek poora round trip).
"""

from llm.client import LLMClient
from raw_agent.tools import TOOL_SCHEMAS, execute_tool

llm = LLMClient()

questions = [
    "What is 1234 * 5678 + 99?",
    "What time is it right now?",
    "Who wrote the Ramayana?",
]

for q in questions:
    print("=" * 60)
    print("USER     :", q)

    resp = llm.chat(
        messages=[{"role": "user", "content": q}],
        tools=TOOL_SCHEMAS,          # model ko "menu card" diya
    )

    if resp.tool_calls:
        # Model ne tool maanga. Yeh sirf ORDER hai, kuch execute nahi hua abhi.
        for tc in resp.tool_calls:
            print("MODEL    : tool call ->", tc["name"], "| args:", tc["arguments"])
            result = execute_tool(tc["name"], tc["arguments"])   # HAMARA code chala raha hai
            print("TOOL RUN :", result)
    else:
        # Model ko laga tool ki zaroorat nahi, isliye seedha text jawab diya.
        print("MODEL    : (no tool) ->", resp.content)

    print("TOKENS   :", resp.usage)
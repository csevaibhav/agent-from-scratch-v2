"""
raw_agent/chat.py — Phase 1, Step 4a: terminal chat with short-term memory
==========================================================================

KYUN BANI:
    Multi-turn baatcheet test karne ke liye. Ek hi Agent object poori chat mein
    rehta hai, isliye uski self.messages (short-term memory) sawaalon ke beech
    bani rehti hai.

COMMANDS:
    /history  -> memory mein kitne messages hain, aur kaun-kaun se roles
    /reset    -> memory saaf (naya thread). LangGraph mein: naya thread_id
    /exit     -> band karo

KAISE CHALAYEIN:
    python -m raw_agent.chat

DHYAN DENA:
    Har turn ke baad "tokens in" badhta jayega, kyunki poori history har baar
    jaati hai. Yahi short-term memory ki keemat hai.
"""

from raw_agent.agent import Agent

agent = Agent(verbose=True)
print("Chat shuru. Commands: /history  /reset  /exit")

while True:
    user = input("\nYOU > ").strip()
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/reset":
        agent.reset()
        print("[memory reset — naya thread]")
        continue
    if user == "/history":
        roles = [m["role"] for m in agent.messages]
        print(f"[{len(agent.messages)} messages] {roles}")
        continue
    
    # Galat/unknown command model tak NA jaaye.
    # Step 4a mein "/history/reset" (typo) model ke paas gaya, aur model ne
    # jhooth bola ki "history reset ho gayi", jabki kuch reset nahi hua tha.
    # Side effects (reset/delete/send) hamesha CODE karta hai, model ka text nahi.
    if user.startswith("/"):
        print("[unknown command] Use: /history  /reset  /exit")
        continue
    
    result = agent.ask(user)
    print("AGENT >", result["answer"])
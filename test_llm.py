"""
test_llm.py — Phase 0 ka smoke test
===================================

KYUN BANI:
    Agent banane se pehle confirm karna hai ki:
      1. Ollama chal raha hai aur qwen2.5:7b model available hai
      2. .env sahi padha ja raha hai
      3. LLMClient wrapper sahi LLMResponse lauta raha hai

KAISE CHALAYEIN:
    python test_llm.py   (project root se, .venv active karke)

KYA DIKHNA CHAHIYE:
    Model ka naam, ek line ka jawab, token usage, aur khaali tool_calls
    (kyunki abhi koi tool nahi bheja).
"""

from llm.client import LLMClient

llm = LLMClient()   # Provider aur model .env se aayenge.

resp = llm.chat([
    # system message model ka "role/behaviour" set karta hai.
    {"role": "system", "content": "You are a helpful assistant. Answer in one line."},
    # user message: asli sawaal.
    {"role": "user", "content": "What is an AI agent?"},
])

print("Model   :", llm.model)
print("Answer  :", resp.content)
print("Usage   :", resp.usage)       # input/output tokens
print("Tools   :", resp.tool_calls)  # Abhi [] aana chahiye
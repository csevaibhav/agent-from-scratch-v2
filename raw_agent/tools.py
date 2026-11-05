"""
raw_agent/tools.py — Agent ke TOOLS: functions + unka JSON schema + dispatcher
==============================================================================

KYUN BANI:
    Agent ke "haath-pair" yahi tools hain. LLM sirf text soch sakta hai;
    calculation, time dekhna jaise kaam tools karte hain.

IS FILE MEIN 4 CHEEZEIN HAIN:
    1. Asli Python functions   -> calculate(), get_current_time()
    2. TOOL_SCHEMAS            -> model ka "menu card" (JSON Schema)
    3. TOOL_REGISTRY           -> tool naam -> Python function
    4. execute_tool()          -> model ke tool_call ko chalane wala dispatcher

STEP 3e REDESIGN (kyun badla):
    Step 3c/3d mein qwen2.5:7b ne:
      - date string "2026-11-04" ko galat parse kiya (20261104 bana diya)
      - "(a - b) * c" ki jagah "a - b * c" bheja (brackets bhool gaya)
    Prompt aur description se 3 baar fix try kiya. Har baar kuch theek hua aur
    kuch toota (regression). Isliye ab TOOL DESIGN badla:
      - get_current_time  -> STRUCTURED JSON (year, day, hour...). Parsing hi khatam.
      - calculate(a, operator, b) -> ek call = EK operation, isliye brackets ki zaroorat nahi.
        operator ek ENUM hai, toh model galat operator bhej hi nahi sakta.
    PRINCIPLE: jo schema se enforce ho sakta hai, use prompt pe mat chhodo.

DESIGN RULES (pehle jaise):
    - Tool hamesha STRING lautata hai (result text ban ke model ke paas jaata hai)
    - Error bhi STRING mein, exception nahi. Agent crash nahi hoga, model sudhaar sakta hai.

LANGCHAIN MAPPING (interview):
    - @tool decorator      = docstring + type hints se yahi schema banata hai
                             (Literal["+", "-", ...] type hint -> enum)
    - model.bind_tools     = tools=TOOL_SCHEMAS bhejna
    - ToolNode (LangGraph) = execute_tool()
"""

import json
import operator
from raw_agent.memory import recall, remember   # Step 4b: long-term memory tools
from datetime import datetime

# ------------------------------------------------------------------
# 1. ASLI FUNCTIONS
# ------------------------------------------------------------------

# Allowed operators. Schema mein bhi exactly yahi ENUM hai.
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod,
}


def calculate(a: float, operator: str, b: float) -> str:
    """EK arithmetic operation: a <operator> b. Multi-step kaam ke liye model ise baar-baar call karega."""
    if operator not in _OPS:                       # enum ke bawajood double-check (defensive coding)
        return f"Error: operator '{operator}' allowed nahi. Use one of {list(_OPS)}"
    try:
        result = _OPS[operator](a, b)
    except ZeroDivisionError:
        return "Error: zero se divide nahi kar sakte"
    # 948.0 ki jagah 948 dikhe, taaki model aur hamara checker dono confuse na ho.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def get_current_time() -> str:
    """Current date/time ko STRUCTURED JSON mein lautata hai, taaki model ko string parse na karni pade."""
    now = datetime.now()
    return json.dumps({
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.strftime("%A"),
        "iso": now.strftime("%Y-%m-%d %H:%M:%S"),   # insaan ke padhne ke liye
    })


# ------------------------------------------------------------------
# 2. TOOL_SCHEMAS — model ka "menu card"
# ------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform exactly ONE arithmetic operation: a <operator> b, "
                           "and return the exact result. Use this for ANY arithmetic. "
                           "For multi-step tasks, call it once per operation, in the order the "
                           "task describes, and use the previous result as the next input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    # ENUM: model sirf inhi mein se chun sakta hai. Schema-level enforcement.
                    "operator": {"type": "string", "enum": list(_OPS),
                                 "description": "The operation to perform"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "operator", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time as JSON with separate fields: "
                           "year, month, day, hour, minute, weekday.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
        {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a durable fact about the user for future conversations "
                           "(their name, a preference, an important personal detail they shared). "
                           "Write it as a short third-person sentence, for example "
                           "'User prefers short answers.' Never save passwords, card numbers or ID numbers.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to save"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search facts saved from EARLIER conversations with the user. "
                           "Call this before answering a question about the user (their name, "
                           "preferences, details they shared before) when it is NOT already in "
                           "the current conversation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string",
                                         "description": "What to look for, e.g. 'name' or 'language preference'"}},
                "required": ["query"],
            },
        },
    },
]

# ------------------------------------------------------------------
# 3. TOOL_REGISTRY — naam se function dhoondhne ki table
# ------------------------------------------------------------------
TOOL_REGISTRY = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "remember": remember,
    "recall": recall,
}


# ------------------------------------------------------------------
# 4. execute_tool — model ka "order" execute karne wala worker
# ------------------------------------------------------------------
def execute_tool(name: str, arguments: str) -> str:
    """
    name      : tool ka naam, jaise "calculate"
    arguments : JSON STRING, jaise '{"a": 2026, "operator": "-", "b": 1947}'
    """
    if name not in TOOL_REGISTRY:
        return f"Error: '{name}' naam ka koi tool nahi hai"
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: arguments valid JSON nahi hain: {arguments}"
    try:
        return TOOL_REGISTRY[name](**args)
    except TypeError as e:                 # galat/missing argument
        return f"Error: galat arguments: {e}"
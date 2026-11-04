"""
raw_agent/tools.py — Agent ke TOOLS: functions + unka JSON schema + dispatcher
==============================================================================

KYUN BANI:
    Agent ke "haath-pair" yahi tools hain. LLM sirf text soch sakta hai.
    Calculation, time dekhna, file padhna jaise kaam tools karte hain.

IS FILE MEIN 4 CHEEZEIN HAIN:
    1. Asli Python functions   -> calculator(), get_current_time()
    2. TOOL_SCHEMAS            -> model ko batane ke liye ki kaunse tools hain
                                  aur unke arguments kya hain (JSON Schema)
    3. TOOL_REGISTRY           -> tool ka naam -> Python function (lookup table)
    4. execute_tool()          -> model ke tool_call ko padh ke sahi function chalata hai

IMPORTANT DESIGN RULES:
    - Tool hamesha STRING lautata hai, kyunki result wapas model ko text ban ke jaata hai.
    - Error bhi STRING mein lautate hain, exception nahi phenkte. Agent crash nahi
      hoga, aur model error padh ke khud galti sudhaar sakta hai.

LANGCHAIN MAPPING (interview):
    - @tool decorator  = function ki docstring + type hints se yahi TOOL_SCHEMAS
                         apne aap bana deta hai
    - model.bind_tools = hamara tools=TOOL_SCHEMAS bhejna
    - ToolNode (LangGraph) = hamara execute_tool()
"""

import ast
import json
import operator
from datetime import datetime


# ------------------------------------------------------------------
# 1. ASLI FUNCTIONS
# ------------------------------------------------------------------

# Calculator mein sirf yeh operators allowed hain.
# eval() use NAHI karte: model ka diya koi bhi Python code (jaise
# file delete karna) chal sakta hai. Yeh security hole hai.
_ALLOWED_OPS = {
    ast.Add: operator.add,      # +
    ast.Sub: operator.sub,      # -
    ast.Mult: operator.mul,     # *
    ast.Div: operator.truediv,  # /
    ast.Pow: operator.pow,      # **
    ast.Mod: operator.mod,      # %
    ast.USub: operator.neg,     # -5 (negative number)
    ast.UAdd: operator.pos,     # +5
}


def _safe_eval(node):
    """Expression ke tree (AST) ko recursively solve karta hai. Sirf numbers aur upar wale operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:      # jaise 2 + 3
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:    # jaise -5
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Sirf numbers aur + - * / ** % allowed hain")


def calculator(expression: str) -> str:
    """Math expression solve karta hai, jaise '1234 * 5678 + 99'."""
    try:
        tree = ast.parse(expression, mode="eval")   # string -> expression tree
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Error: {e}"                        # error bhi string mein, crash nahi


def get_current_time() -> str:
    """Laptop ka current date aur time lautata hai."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


# ------------------------------------------------------------------
# 2. TOOL_SCHEMAS — model ko tools ka "menu card"
# ------------------------------------------------------------------
# Format OpenAI ka function-calling format hai (Ollama bhi yahi samajhta hai).
# DHYAN: "description" asal mein PROMPT hai. Model isi ko padh ke decide
# karta hai ki tool kab use karna hai. Description kharab ho toh tool choice galat hogi.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            # DESCRIPTION = tool ka prompt. Step 3c mein model ne "(a - b) * c" ki jagah
            # "a - b * c" bheja (brackets missing) aur jawab galat aaya.
            # "KAISE use karna hai" wale rules yahin likhte hain, system prompt mein nahi.
            # Example NEUTRAL hai ('(10 - 3) * 5'), test sawaalon jaisa nahi (leakage se bachav).
            "description": "Evaluate a math expression and return the exact result. "
                           "Use this for ANY arithmetic instead of calculating yourself. "
                           "IMPORTANT: Python operator precedence applies (* and / before + and -). "
                           "When a task says to do one operation and THEN another, wrap the earlier "
                           "part in parentheses so the order is preserved. "
                           "Example: 'subtract 3 from 10, then multiply by 5' -> '(10 - 3) * 5', "
                           "NOT '10 - 3 * 5'.",
            "parameters": {                      # JSON Schema: kaunse arguments, kis type ke
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using numbers and + - * / ** %",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date, time and day of the week.",
            "parameters": {"type": "object", "properties": {}, "required": []},  # koi argument nahi
        },
    },
]


# ------------------------------------------------------------------
# 3. TOOL_REGISTRY — naam se function dhoondhne ki table
# ------------------------------------------------------------------
# Model sirf tool ka NAAM (string) bhejta hai. Yeh dict us naam ko
# asli Python function se jodti hai.
TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
}


# ------------------------------------------------------------------
# 4. execute_tool — "worker" jo model ka order execute karta hai
# ------------------------------------------------------------------
def execute_tool(name: str, arguments: str) -> str:
    """
    name      : model ne jo tool maanga, jaise "calculator"
    arguments : JSON STRING, jaise '{"expression": "2+2"}'
                (model string bhejta hai, dict nahi, isliye json.loads karna padta hai)
    """
    if name not in TOOL_REGISTRY:                   # model kabhi galat/naya tool naam bana deta hai
        return f"Error: '{name}' naam ka koi tool nahi hai"

    try:
        args = json.loads(arguments) if arguments else {}   # string -> dict
    except json.JSONDecodeError:
        return f"Error: arguments valid JSON nahi hain: {arguments}"

    try:
        return TOOL_REGISTRY[name](**args)          # **args: dict ko keyword arguments bana deta hai
    except TypeError as e:                          # galat/missing argument naam
        return f"Error: galat arguments: {e}"
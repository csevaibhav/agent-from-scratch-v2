"""
llm/client.py — Provider-agnostic LLM wrapper
==============================================

KYUN BANI:
    Agent ka code kabhi seedha OpenAI / Ollama / Anthropic / Bedrock
    ko call nahi karega. Woh sirf `LLMClient.chat()` ko call karega.
    Provider badalna ho (jaise Phase 9 mein AWS Bedrock), toh sirf
    yeh file badlegi. Agent loop, tools, memory sab waise hi rahenge.
    (Decision D5)

KYA KARTI HAI:
    1. .env se settings padhti hai (provider, model, URL).
    2. Messages (+ optional tools) leke LLM ko call karti hai.
    3. Provider ka response ek COMMON shape mein badal deti hai,
       yaani `LLMResponse(content, tool_calls, usage)`.

KAUN USE KARTA HAI:
    test_llm.py (abhi), aur Phase 1 se raw agent loop.

ANALOGY:
    Mobile charger ka adapter. Socket (provider) koi bhi ho,
    phone (agent) ko hamesha same pin milti hai.

INTERVIEW POINT:
    "LangChain ka ChatModel / AIMessage abstraction exactly yahi
    normalization layer hai. Har provider ka alag response ek common
    message object mein. Maine pehle khud likh ke samjha ki
    framework andar kya karta hai."
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv   # .env file ki values ko environment variables mein daalta hai
from openai import OpenAI        # OpenAI ka SDK. Ollama bhi same API format bolta hai.

# Program start hote hi .env padh lo, taaki os.getenv() ko values mil jaayein.
load_dotenv()


@dataclass
class LLMResponse:
    """
    Har provider ka jawab is EK shape mein aayega.
    Agent ko sirf yeh 4 fields dikhenge, provider ka asli format kabhi nahi.
    """
    content: str | None                              # Model ka text jawab. Tool call ke time None ho sakta hai.
    tool_calls: list = field(default_factory=list)   # Model ne jo tools call karne ko kaha: [{"id", "name", "arguments"}]
    usage: dict = field(default_factory=dict)         # Kitne tokens gaye/aaye. Phase 2 traces aur cost ke liye.
    raw: object = None                                # Provider ka original response. Sirf debugging ke liye.


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        # Argument diya toh woh use hoga, warna .env se, warna default "ollama".
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")

        if self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
            # Ollama OpenAI-compatible endpoint deta hai, isliye OpenAI SDK
            # ko bas base_url badal ke Ollama ki taraf mod dete hain.
            self._client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",  # Ollama key check nahi karta, par SDK ko koi value chahiye.
            )
        else:
            # Anthropic / Bedrock aage ke phases mein yahan add honge.
            raise ValueError(f"Provider '{self.provider}' abhi supported nahi hai")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0) -> LLMResponse:
        """
        messages    : conversation history, jaise [{"role": "user", "content": "..."}]
        tools       : model ko available tools ka JSON schema (Phase 1 se use hoga)
        temperature : 0 = deterministic. Same input pe same tool decision aayega,
                      jo debugging aur evals ke liye zaroori hai.
        """
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools   # Tools sirf tab bhejo jab ho. Khaali list kuch models ko confuse karti hai.

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message   # OpenAI format: jawab choices[0].message mein hota hai.

        # NORMALIZATION: OpenAI ke tool_call objects ko simple dicts mein badlo.
        # "arguments" abhi JSON STRING hai. Usse dict mein badalna Phase 1 mein agent karega.
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]

        # Token count: provider "prompt/completion" bolta hai, hum apne naam rakhte hain.
        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}

        return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage, raw=resp)
"""
llm/client.py — Provider-agnostic LLM wrapper (Ollama + Anthropic)
==================================================================

KYUN BANI:
    Agent ka code kabhi seedha kisi provider ko call nahi karta, sirf
    LLMClient.chat() ko karta hai. Provider badalna = sirf yeh file. (D5)

DESIGN (D8): CANONICAL FORMAT + ADAPTERS
    Agent ki history HAMESHA OpenAI format mein rehti hai:
        {"role": "system"|"user"|"assistant"|"tool", ...}
    Har provider ke liye yahan ek ADAPTER hai jo is format ko us provider
    ke format mein badalta hai, aur jawab ko wapas LLMResponse mein.
      - Ollama    : already OpenAI-compatible, conversion nahi chahiye
      - Anthropic : system alag param, tool calls "tool_use" blocks,
                    tool results "tool_result" blocks ek USER message mein

PROVIDERS:
    ollama    -> local models (qwen2.5:7b, qwen3:8b)
    anthropic -> Claude (API key .env mein: ANTHROPIC_API_KEY)

INTERVIEW POINT:
    "LangChain ke HumanMessage / AIMessage / ToolMessage canonical messages hain,
    aur har provider integration unhe apne format mein convert karti hai.
    Maine yahi adapter pattern khud likha, OpenAI-style aur Anthropic-style
    tool calling ke beech."
"""

import json
import urllib.error
import urllib.request
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMResponse:
    """Har provider ka jawab is EK shape mein aata hai. Agent sirf yeh dekhta hai."""
    content: str | None                              # text jawab (tool call pe None/"" ho sakta hai)
    tool_calls: list = field(default_factory=list)   # [{"id", "name", "arguments" (JSON string)}]
    usage: dict = field(default_factory=dict)         # {"input_tokens", "output_tokens"}
    raw: object = None                                # provider ka original response (debugging)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")

        if self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
            self._client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",   # Ollama key check nahi karta
            )
        elif self.provider == "anthropic":
            # SDK nahi, seedha HTTP (urllib). Anthropic SDK ki "jiter" DLL ko Windows
            # Application Control ne block kiya tha. urllib Python mein built-in hai,
            # isliye koi extra DLL nahi lagti. SDK asal mein HTTP ke upar wrapper hi hai.
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
            self._api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self._api_key:
                raise ValueError("ANTHROPIC_API_KEY .env mein nahi mili")

    # ==================================================================
    # PUBLIC: agent sirf yeh method call karta hai
    # ==================================================================
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0) -> LLMResponse:
        """messages aur tools hamesha OpenAI (canonical) format mein aate hain."""
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, temperature)
        return self._chat_openai_compatible(messages, tools, temperature)

    # ==================================================================
    # ADAPTER 1: OpenAI-compatible (Ollama)
    # ==================================================================
    def _chat_openai_compatible(self, messages, tools, temperature) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        content = msg.content
        # qwen3 jaise "thinking" models apni soch <think>...</think> mein likh dete hain.
        # Yeh final jawab nahi hai, toh hata do, warna history aur answer dono ganda hoga.
        if content and "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage, raw=resp)

    # ==================================================================
    # ADAPTER 2: Anthropic (Claude)
    # ==================================================================
    def _chat_anthropic(self, messages, tools, temperature) -> LLMResponse:
        # (1) System prompt Anthropic mein ALAG parameter hai, messages ka hissa nahi.
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")

        # (2) Canonical (OpenAI) history -> Anthropic format
        conv = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            if role == "user":
                conv.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                # Anthropic mein assistant ka content = BLOCKS ki list (text + tool_use)
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"] or "{}"),  # Anthropic dict maangta hai, string nahi
                    })
                if blocks:   # Anthropic khaali content reject karta hai
                    conv.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                # Anthropic mein tool result = USER message ke andar "tool_result" block.
                # Ek step ke saare tool results EK hi user message mein jaane chahiye.
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                    conv[-1]["content"].append(block)
                else:
                    conv.append({"role": "user", "content": [block]})

                # NOTE: temperature yahan NAHI bhejte. Naye Claude models pe yeh param deprecated hai
        # (API 400 error deta hai). Har provider/model ke supported params alag hote hain.
        # Yeh farak sambhaalna adapter ka kaam hai, agent ka nahi. (D8)
        # Randomness control nahi hogi, isliye evals multiple runs + pass RATE pe chalte hain.
        kwargs = {"model": self.model, "max_tokens": 1024, "system": system,
                  "messages": conv}

        # (3) Tool schema: OpenAI {"function": {"parameters": ...}} -> Anthropic {"input_schema": ...}
        if tools:
            kwargs["tools"] = [
                {"name": t["function"]["name"],
                 "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]}
                for t in tools
            ]

        data = self._anthropic_post(kwargs)

        # (4) Anthropic response (JSON dict) -> LLMResponse (canonical)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text") or None
        tool_calls = [
            {"id": b["id"], "name": b["name"], "arguments": json.dumps(b["input"])}   # wapas JSON string
            for b in data["content"] if b["type"] == "tool_use"
        ]
        usage = {"input_tokens": data["usage"]["input_tokens"],
                 "output_tokens": data["usage"]["output_tokens"]}
        return LLMResponse(content=text, tool_calls=tool_calls, usage=usage, raw=data)

    def _anthropic_post(self, payload: dict) -> dict:
        """
        Anthropic Messages API ko SEEDHA HTTP POST. SDK bhi andar yahi karta hai:
          URL     : https://api.anthropic.com/v1/messages
          Headers : x-api-key (auth), anthropic-version (API version), content-type
          Body    : model, max_tokens, system, messages, tools (JSON)
        """
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # API ka error message dikhao (jaise galat model naam / key). Key kabhi print nahi hoti.
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"Anthropic API error {e.code}: {body}") from None

    # ==================================================================
    # MESSAGE BUILDERS: canonical (OpenAI) format mein history messages
    # ==================================================================
    def assistant_message(self, resp: LLMResponse) -> dict:
        """Model ka response history mein daalne layak message (canonical format)."""
        msg = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ]
        return msg

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """Tool ka result history mein (canonical format)."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
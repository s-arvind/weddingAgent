import os
import re
import json
import logging
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from app.agent.tools import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

MAX_TURNS      = int(os.getenv("AGENT_MAX_TURNS", 5))
MAX_TOOL_CHARS = int(os.getenv("AGENT_MAX_TOOL_CHARS", 2000))

# llama-3.3 on Groq occasionally emits tool calls as text in the content field
# (e.g. <function=search_vendors>{"query": "..."}</function>) instead of as a
# structured tool_call. Detect and recover these so the search still runs.
_LEAKED_CALL_RE = re.compile(r"<function=([a-zA-Z_]\w*)>\s*(\{.*?\})\s*</function>", re.DOTALL)


def _extract_leaked_tool_calls(content: str) -> list[dict]:
    calls = []
    for name, raw_args in _LEAKED_CALL_RE.findall(content):
        try:
            calls.append({"name": name, "arguments": json.loads(raw_args)})
        except json.JSONDecodeError:
            continue
    return calls


def _strip_leaked_syntax(content: str) -> str:
    """Remove any leaked/unparseable function-call tags so the user never sees raw syntax."""
    content = _LEAKED_CALL_RE.sub("", content)
    content = re.sub(r"</?function[^>]*>", "", content)
    return content.strip()

SYSTEM_PROMPT = """You are Knot — an AI wedding planner with deep knowledge of Indian weddings and access to a real vendor database across India.

You think and respond like an experienced planner who has worked on hundreds of weddings — warm, direct, practical. Hinglish is fine. You build on everything said earlier in the conversation.

You help with the full wedding journey: planning timelines, understanding rituals and traditions (Hindu, Muslim, Sikh, regional variations), budgeting, and finding the right vendors — venues, caterers, photographers, decorators, makeup artists, pandits, DJs, and more.

TOOLS:
- search_vendors — search by type, city, guest count, budget
- get_vendor_details — full profile, contact, address for a specific vendor
- compare_vendors — side-by-side on price, capacity, occasions
- find_similar_vendors — alternatives to a vendor already shown
- estimate_budget — real cost breakdown from database by city, guest count, categories

Use tools whenever the user needs vendor information — including follow-ups like "contact details", "address", "phone number", or "tell me more about that one". The full tool results including vendor slugs are in your context — use them.

For venues and caterers, confirm the city and rough guest count before searching — without them the results aren't useful. Once you have what you need, run the search directly rather than only offering to. When the user asks for more options, refine the search with different filters rather than repeating the same query. If a search comes back empty, broaden the filters or ask the user to loosen a constraint — never fill the gap with invented vendors.

When presenting vendors: include name, city, pricing, capacity (if relevant), and one line on what makes them stand out. Extract actual package tiers for photographers/decorators rather than quoting raw text. Skip fields with no data.

GUARDRAILS:
- Only use what tools return — never invent vendor names, prices, or contact details.
- Search results are the top matches by relevance, not every vendor in the city. Present them as "some good options," never as the complete list — offer to search differently if they want more.
- Prices always in Rs.
- Stay focused on weddings and wedding planning.
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        raw = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        _client = wrap_openai(raw)
    return _client


def chat(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Run one full agentic turn.
    Returns (final_answer, updated_user_assistant_messages).
    """
    client = _get_client()
    model  = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            leaked = _extract_leaked_tool_calls(content)
            if leaked:
                # model wrote the tool call as text — execute it and loop again
                full_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"leaked_{i}",
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                        }
                        for i, c in enumerate(leaked)
                    ],
                })
                for i, c in enumerate(leaked):
                    result = dispatch(c["name"], c["arguments"])
                    if len(result) > MAX_TOOL_CHARS:
                        result = result[:MAX_TOOL_CHARS] + "... [truncated]"
                    logger.warning("Recovered leaked tool %s args=%s result_len=%d",
                                   c["name"], c["arguments"], len(result))
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": f"leaked_{i}",
                        "content": result,
                    })
                continue  # re-prompt so the model answers using the tool results

            answer = _strip_leaked_syntax(content)
            full_messages.append({"role": "assistant", "content": answer})
            return answer, full_messages[1:]  # exclude system message; keep tool messages for context

        full_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = dispatch(tc.function.name, args)
            # truncate large tool results to stay within context window
            if len(result) > MAX_TOOL_CHARS:
                result = result[:MAX_TOOL_CHARS] + "... [truncated]"
            logger.info("Tool %s args=%s result_len=%d", tc.function.name, args, len(result))
            full_messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # max turns hit — ask LLM to summarize what it found
    full_messages.append({
        "role":    "user",
        "content": "Please summarize what you found so far in a helpful response.",
    })
    summary = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.2,
    )
    answer  = summary.choices[0].message.content or "I was unable to complete your request."
    full_messages.append({"role": "assistant", "content": answer})
    return answer, full_messages[1:]

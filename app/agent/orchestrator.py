import os
import json
import logging
from openai import OpenAI
from app.agent.tools import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", 5))

SYSTEM_PROMPT = """You are a helpful wedding planning assistant for India.
You help users find vendors — venues, caterers, photographers, pandits, decorators, and more.
You understand Hinglish queries (mixed Hindi-English) and Indian wedding ceremonies like
baarat, haldi, mehndi, sangeet, ring ceremony.

STRICT RULES:
- Always use search_vendors tool before answering any vendor question.
- Only recommend vendors that appear in tool results. NEVER suggest vendors not in the results.
- If no vendors found, say "No vendors found in our database for this query" and suggest refining the search.
- Be specific about pricing in ₹, capacity, and location from the tool results.
- If user asks to compare, use compare_vendors tool.
- If user wants details about a specific vendor, use get_vendor_details tool.
"""


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


def chat(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Run one full agentic turn.
    Returns (final_answer, updated_messages).
    """
    client = _get_client()
    model  = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message

        # no tool call → final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            messages = messages + [{"role": "assistant", "content": answer}]
            return answer, messages

        # execute tool calls
        full_messages.append(msg)
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = dispatch(tc.function.name, args)
            logger.info("Tool %s args=%s result_len=%d",
                        tc.function.name, args, len(result))
            full_messages.append({
                "role":        "tool",
                "tool_call_id": tc.id,
                "content":     result,
            })

    # max turns hit — return last content
    last = full_messages[-1].get("content", "I could not complete the request.")
    messages = messages + [{"role": "assistant", "content": last}]
    return last, messages

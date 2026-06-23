import os
import json
import logging
from openai import OpenAI
from app.agent.tools import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

MAX_TURNS      = int(os.getenv("AGENT_MAX_TURNS", 5))
MAX_TOOL_CHARS = int(os.getenv("AGENT_MAX_TOOL_CHARS", 2000))

SYSTEM_PROMPT = """You are an expert Indian wedding planning assistant. You help couples plan their dream wedding.

You have two roles:
1. WEDDING ADVISOR — Answer general wedding planning questions from your knowledge:
   - Food menus, guest lists, ceremonies, traditions, budgeting, timelines
   - What to wear, which rituals to follow, Hinglish terms (baarat, haldi, mehndi, sangeet, etc.)
   - Do NOT use tools for these — answer directly and helpfully.

2. VENDOR FINDER — Find real vendors from our database when users need:
   - Venues, caterers, photographers, pandits, decorators, makeup artists, etc.
   - Use search_vendors tool to find them. Only recommend vendors from tool results.
   - If no vendors found, say so and suggest refining the search.
   - Use get_vendor_details for more info on a specific vendor.
   - Use compare_vendors to compare multiple vendors.
   - Use estimate_budget when user asks about cost or budget for their wedding.
   - Use find_similar_vendors when user says "show me more like this" or wants alternatives.

RULES:
- Never hallucinate vendor names — only use results from tools.
- For general advice questions, answer directly without calling any tool.
- Be warm, helpful, and specific. Use Rs for prices.
- Understand Hinglish naturally.
- If the question is NOT related to weddings, wedding planning, or wedding vendors, politely decline and say you can only help with wedding-related topics.
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
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
            answer = msg.content or ""
            updated = messages + [{"role": "assistant", "content": answer}]
            return answer, updated

        full_messages.append(msg)
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
    updated = messages + [{"role": "assistant", "content": answer}]
    return answer, updated

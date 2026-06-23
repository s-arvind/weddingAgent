import uuid
import logging
from openai import BadRequestError
from fastapi import APIRouter
from app.api.schemas import ChatRequest, ChatResponse, ChatMessage, HealthResponse
from app.agent.orchestrator import chat
from app.db.postgres import check_health as pg_health
from app.db.qdrant import check_health as qdrant_health
from app.api.session import get_history, save_history

logger = logging.getLogger(__name__)

router = APIRouter()

FALLBACK_ANSWER = (
    "I'm a wedding vendor assistant for India. "
    "I can help you find venues, caterers, photographers, pandits, and more. "
    "Try asking something like: 'Find a wedding venue in Mumbai for 200 guests'."
)


@router.get("/health", response_model=HealthResponse)
def health():
    pg  = pg_health()
    qdr = qdrant_health()
    return HealthResponse(
        status="ok" if pg and qdr else "degraded",
        postgres=pg,
        qdrant=qdr,
    )


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history    = get_history(session_id)
    history.append({"role": "user", "content": req.message})

    try:
        answer, updated = chat(history)
    except BadRequestError as e:
        logger.warning("LLM bad request (off-topic or tool gen failure): %s", e)
        answer  = FALLBACK_ANSWER
        updated = history + [{"role": "assistant", "content": answer}]
    except Exception as e:
        logger.error("Agent error: %s", e)
        answer  = "Something went wrong. Please try again."
        updated = history + [{"role": "assistant", "content": answer}]

    # strip tool-role messages — they break Groq on replay and confuse clients
    safe_history = [m for m in updated if m["role"] in ("user", "assistant")]
    save_history(session_id, safe_history)

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        messages=[ChatMessage(role=m["role"], content=m["content"])
                  for m in safe_history],
    )

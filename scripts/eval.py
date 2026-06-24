#!/usr/bin/env python3
"""
RAGAS evaluation suite for Wedding Agent RAG pipeline.
Targets: answer_relevancy > 0.80, context_precision > 0.75

Usage:
    python scripts/eval.py
    python scripts/eval.py --output results/eval_20240101.json
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import datasets
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    {"question": "Find a wedding venue in Mumbai for 200 guests with vegetarian food"},
    {"question": "Wedding photographer in Hyderabad under 50000 rupees"},
    {"question": "Pandit for haldi and mehndi ceremony in Delhi"},
    {"question": "Caterers for sangeet and baarat in Jaipur"},
    {"question": "Budget for a 300 guest wedding in Bangalore"},
    {"question": "Makeup artist for bridal lehenga look in Chennai"},
    {"question": "Outdoor wedding venues in Udaipur for 500 guests"},
    {"question": "Vegetarian catering for wedding reception in Pune"},
    {"question": "Best photographers for candid wedding photos in Kolkata"},
    {"question": "Decorator for mehndi night ceremony in Ahmedabad"},
    {"question": "Wedding venue with accommodation in Goa for 150 guests"},
    {"question": "South Indian wedding caterers in Hyderabad"},
    {"question": "Flower decoration for mandap in Delhi under 1 lakh"},
    {"question": "DJ for sangeet night in Mumbai"},
    {"question": "Horse and barat services in Jaipur for wedding procession"},
    {"question": "Find a pandit for Gujarati wedding rituals in Surat"},
    {"question": "Tent house and lighting for outdoor wedding in Lucknow"},
    {"question": "Pre-wedding shoot photographers in Shimla"},
    {"question": "Catering for 500 guests in Delhi with both veg and non-veg options"},
    {"question": "Bridal mehendi artist in Hyderabad for full hands and feet"},
    {"question": "5-star hotel venue for wedding in Chennai for 300 guests"},
    {"question": "Budget wedding packages in Indore under 5 lakhs"},
    {"question": "Compare top catering services in Bangalore"},
    {"question": "Find similar venues to a hall in Mumbai I already liked"},
    {"question": "What is the average cost of a wedding venue in Delhi"},
]


class ProjectEmbeddings(Embeddings):
    """Wraps the project's BGE-M3 embedder in a LangChain-compatible interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.rag.embedder import embed
        vecs = embed(texts)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        from app.rag.embedder import embed
        return embed([text])[0].tolist()


def build_dataset(queries: list[dict]) -> datasets.Dataset:
    from app.rag.retriever import retrieve
    from app.agent.orchestrator import chat as agent_chat

    rows = []
    for i, item in enumerate(queries):
        q = item["question"]
        logger.info("[%d/%d] Running query: %s", i + 1, len(queries), q)

        chunks = retrieve(q, final_top_n=5)
        contexts = [c.text for c in chunks if c.text]

        history = [{"role": "user", "content": q}]
        try:
            answer, _ = agent_chat(history)
        except Exception as e:
            logger.warning("Agent failed for '%s': %s", q, e)
            answer = ""

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
        })

    return datasets.Dataset.from_list(rows)


def run_eval(output_path: str | None = None) -> dict:
    groq_llm = LangchainLLMWrapper(ChatOpenAI(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(ProjectEmbeddings())

    logger.info("Building evaluation dataset (%d queries)...", len(SAMPLE_QUERIES))
    dataset = build_dataset(SAMPLE_QUERIES)

    metrics = [
        answer_relevancy,
        context_precision,
    ]
    for m in metrics:
        m.llm = groq_llm
    answer_relevancy.embeddings = embeddings

    logger.info("Running RAGAS evaluation...")
    results = evaluate(dataset, metrics=metrics)

    scores = {
        "answer_relevancy":  round(float(results["answer_relevancy"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
    }

    print("\n── RAGAS Evaluation Results ──────────────────────")
    print(f"  answer_relevancy  : {scores['answer_relevancy']:.4f}  (target > 0.80)")
    print(f"  context_precision : {scores['context_precision']:.4f}  (target > 0.75)")
    print("──────────────────────────────────────────────────\n")

    passed = (
        scores["answer_relevancy"] > 0.80
        and scores["context_precision"] > 0.75
    )
    print("PASSED" if passed else "BELOW TARGET — review retrieval and prompting")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"scores": scores, "passed": passed, "n_queries": len(SAMPLE_QUERIES)}
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results written to %s", output_path)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Wedding Agent")
    parser.add_argument("--output", help="Path to write JSON results (optional)")
    args = parser.parse_args()
    run_eval(output_path=args.output)

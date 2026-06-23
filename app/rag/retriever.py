import os
import logging
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from app.rag.embedder import embed
from app.models.vendor_collection import VendorCollection

logger = logging.getLogger(__name__)

TOP_K      = int(os.getenv("RETRIEVER_TOP_K", 20))
RRF_K      = int(os.getenv("RETRIEVER_RRF_K", 60))
FINAL_TOP_N = int(os.getenv("RETRIEVER_FINAL_TOP_N", 10))


@dataclass
class RetrievedChunk:
    chunk_id:    str
    vendor_id:   str
    vendor_slug: str
    vendor_name: str
    city:        str
    state:       str
    chunk_type:  str
    text:        str
    score:       float
    categories:  list[str]
    capacity_min: int | None
    capacity_max: int | None
    price_veg_min: int | None
    price_veg_max: int | None
    price_nonveg_min: int | None
    price_nonveg_max: int | None


def _build_qdrant_filter(filters: dict) -> Filter | None:
    conditions = []

    if city := filters.get("city"):
        conditions.append(FieldCondition(key="city", match=MatchValue(value=city)))

    if state := filters.get("state"):
        conditions.append(FieldCondition(key="state", match=MatchValue(value=state)))

    if chunk_type := filters.get("chunk_type"):
        conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))

    if cap := filters.get("capacity_min"):
        conditions.append(FieldCondition(key="capacity_max", range=Range(gte=cap)))

    if price := filters.get("price_veg_max"):
        conditions.append(FieldCondition(key="price_veg_min", range=Range(lte=price)))

    return Filter(must=conditions) if conditions else None


def _rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(
    query: str,
    filters: dict | None = None,
    top_k: int = TOP_K,
    final_top_n: int = FINAL_TOP_N,
) -> list[RetrievedChunk]:
    filters = filters or {}

    # ── dense retrieval ──────────────────────────────────────────────────────
    query_vector = embed([query])[0]
    qdrant_filter = _build_qdrant_filter(filters)

    dense_hits = VendorCollection.search(
        vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
    )

    # build chunk map for RRF + final output
    chunk_map: dict[str, dict] = {}
    for hit in dense_hits:
        p = hit.payload
        chunk_map[p["chunk_id"]] = {
            "chunk_id":       p["chunk_id"],
            "vendor_id":      p.get("vendor_id", ""),
            "vendor_slug":    p.get("vendor_slug", ""),
            "vendor_name":    p.get("vendor_name", ""),
            "city":           p.get("city", ""),
            "state":          p.get("state", ""),
            "chunk_type":     p.get("chunk_type", ""),
            "text":           p.get("text", ""),
            "categories":     p.get("categories", []),
            "capacity_min":   p.get("capacity_min"),
            "capacity_max":   p.get("capacity_max"),
            "price_veg_min":  p.get("price_veg_min"),
            "price_veg_max":  p.get("price_veg_max"),
            "price_nonveg_min": p.get("price_nonveg_min"),
            "price_nonveg_max": p.get("price_nonveg_max"),
        }

    dense_ranking = [p["chunk_id"] for hit in dense_hits for p in [hit.payload]]

    # ── sparse retrieval (BM25) ───────────────────────────────────────────────
    corpus = [cid.replace("-", " ").replace("_", " ").split() for cid in chunk_map]
    bm25_ranking: list[str] = []
    if corpus:
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        chunk_ids = list(chunk_map.keys())
        bm25_ranking = [chunk_ids[i] for i in sorted(range(len(scores)),
                        key=lambda x: scores[x], reverse=True)]

    # ── RRF fusion ───────────────────────────────────────────────────────────
    fused = _rrf([dense_ranking, bm25_ranking])[:final_top_n]

    results = []
    for chunk_id in fused:
        if chunk_id not in chunk_map:
            continue
        c = chunk_map[chunk_id]
        results.append(RetrievedChunk(
            chunk_id=c["chunk_id"],
            vendor_id=c["vendor_id"],
            vendor_slug=c["vendor_slug"],
            vendor_name=c["vendor_name"],
            city=c["city"],
            state=c["state"],
            chunk_type=c["chunk_type"],
            text=c["text"],
            score=0.0,
            categories=c["categories"],
            capacity_min=c["capacity_min"],
            capacity_max=c["capacity_max"],
            price_veg_min=c["price_veg_min"],
            price_veg_max=c["price_veg_max"],
            price_nonveg_min=c["price_nonveg_min"],
            price_nonveg_max=c["price_nonveg_max"],
        ))

    logger.info("Retrieved %d chunks for query: %s", len(results), query[:60])
    return results

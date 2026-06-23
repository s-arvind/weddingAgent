import os
import logging
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    OptimizersConfigDiff, HnswConfigDiff,
    PayloadSchemaType, SearchRequest,
    Filter,
)
from app.db.qdrant import get_client

logger = logging.getLogger(__name__)


class VendorCollection:
    NAME              = "vendors"
    DIM               = int(os.getenv("EMBEDDING_DIM", 1024))
    UPSERT_BATCH      = 256
    REPLICATION_FACTOR = int(os.getenv("QDRANT_REPLICATION_FACTOR", 1))

    @classmethod
    def create(cls) -> None:
        client = get_client()
        existing = {c.name for c in client.get_collections().collections}
        if cls.NAME in existing:
            logger.info("Qdrant collection '%s' already exists", cls.NAME)
            return

        try:
            client.create_collection(
                collection_name=cls.NAME,
                vectors_config=VectorParams(
                    size=cls.DIM,
                    distance=Distance.COSINE,
                    on_disk=True,              # vectors stored on disk — scales beyond RAM
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10_000,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
                replication_factor=cls.REPLICATION_FACTOR,
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Qdrant collection '%s' created by another worker", cls.NAME)
                return
            raise

        for field in ["city", "state", "chunk_type"]:
            client.create_payload_index(cls.NAME, field, PayloadSchemaType.KEYWORD)

        for field in ["capacity_min", "capacity_max",
                      "price_veg_min", "price_veg_max",
                      "price_nonveg_min", "price_nonveg_max"]:
            client.create_payload_index(cls.NAME, field, PayloadSchemaType.INTEGER)

        logger.info(
            "Qdrant collection '%s' created (replication_factor=%d)",
            cls.NAME, cls.REPLICATION_FACTOR,
        )

    @classmethod
    def upsert(cls, points: list[PointStruct]) -> None:
        if not points:
            return
        client = get_client()
        for i in range(0, len(points), cls.UPSERT_BATCH):
            batch = points[i : i + cls.UPSERT_BATCH]
            client.upsert(collection_name=cls.NAME, points=batch, wait=True)
        logger.info("Upserted %d points to Qdrant", len(points))

    @classmethod
    def search(
        cls,
        vector: list[float],
        query_filter: Filter | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list:
        result = get_client().query_points(
            collection_name=cls.NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return result.points

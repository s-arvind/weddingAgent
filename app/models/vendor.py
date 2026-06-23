import logging
from sqlalchemy import Column, Text, Integer, Numeric, ARRAY, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Vendor(Base):
    __tablename__ = "vendors"

    id               = Column(Text, primary_key=True)
    name             = Column(Text)
    address          = Column(Text)
    city             = Column(Text)
    state            = Column(Text)
    lat              = Column(Numeric(10, 6))
    lng              = Column(Numeric(10, 6))
    mobile           = Column(Text)
    categories       = Column(ARRAY(Text))
    occasions        = Column(ARRAY(Text))
    payment_methods  = Column(ARRAY(Text))
    capacity_min     = Column(Integer)
    capacity_max     = Column(Integer)
    price_veg_min    = Column(Integer)
    price_veg_max    = Column(Integer)
    price_nonveg_min = Column(Integer)
    price_nonveg_max = Column(Integer)
    image_urls       = Column(ARRAY(Text))
    raw_description  = Column(Text)
    raw_faqs         = Column(JSONB)
    ingested_at      = Column(TIMESTAMP(timezone=True), server_default=func.now())

    @classmethod
    def bulk_insert(cls, session: Session, vendors: list[dict]) -> None:
        if not vendors:
            return
        stmt = insert(cls).values(vendors)
        stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
        session.execute(stmt)
        logger.info(f"Inserted {len(vendors)} vendors")

    @classmethod
    def get_by_id(cls, session: Session, vendor_id: str) -> "Vendor | None":
        return session.get(cls, vendor_id)

    @classmethod
    def get_by_ids(cls, session: Session, vendor_ids: list[str]) -> list["Vendor"]:
        return session.query(cls).filter(cls.id.in_(vendor_ids)).all()

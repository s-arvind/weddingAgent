import json
import logging
from contextlib import contextmanager
from sqlalchemy import func, and_
from sqlalchemy.orm import Session
from app.db.postgres import get_engine
from app.models.vendor import Vendor
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


@contextmanager
def _session():
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── tool schemas (sent to Groq) ───────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_vendors",
            "description": (
                "Search wedding vendors using natural language. "
                "Use for queries like 'venues in Mumbai for 200 guests' or "
                "'pandit for haldi ceremony in Delhi'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":         {"type": "string",  "description": "Natural language search query"},
                    "city":          {"type": "string",  "description": "Filter by city name"},
                    "state":         {"type": "string",  "description": "Filter by state name"},
                    "capacity_min":  {"type": "integer", "description": "Minimum guest capacity needed"},
                    "price_veg_max": {"type": "integer", "description": "Max veg price per plate in INR"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vendor_details",
            "description": "Get full details of a specific vendor by their slug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_slug": {"type": "string", "description": "Vendor slug identifier"},
                },
                "required": ["vendor_slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_vendors",
            "description": "Compare multiple vendors side by side on price, capacity, and occasions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_slugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of vendor slugs to compare",
                    },
                },
                "required": ["vendor_slugs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": (
                "Estimate wedding budget based on guest count, city, and required vendor categories. "
                "Pulls real price data from the vendor database to give min/avg/max per category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "guest_count": {"type": "integer", "description": "Number of wedding guests"},
                    "city":        {"type": "string",  "description": "Wedding city"},
                    "categories":  {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Vendor categories needed, e.g. ['venue', 'caterer', 'photographer']",
                    },
                },
                "required": ["guest_count", "city", "categories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_vendors",
            "description": (
                "Find vendors similar to a given vendor — same city, overlapping categories, "
                "similar price range. Use when user says 'show me more like this' or wants alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_slug": {"type": "string",  "description": "Slug of the reference vendor"},
                    "limit":       {"type": "integer", "description": "Max results to return (default 5)"},
                },
                "required": ["vendor_slug"],
            },
        },
    },
]


# ── tool implementations ──────────────────────────────────────────────────────

def search_vendors(
    query: str,
    city: str | None = None,
    state: str | None = None,
    capacity_min: int | None = None,
    price_veg_max: int | None = None,
) -> str:
    filters: dict = {}
    if city is not None:          filters["city"] = city
    if state is not None:         filters["state"] = state
    if capacity_min is not None:  filters["capacity_min"] = capacity_min
    if price_veg_max is not None: filters["price_veg_max"] = price_veg_max

    chunks = retrieve(query, filters=filters, final_top_n=20)

    # group all chunks per vendor, preserving RRF rank order
    vendor_chunks: dict[str, list] = {}
    for c in chunks:
        vendor_chunks.setdefault(c.vendor_slug, []).append(c)

    PRICING_KEYWORDS = ("package", "price", "cost", "charges", "fee", "rate", "starting")

    vendors = []
    for slug, vchunks in list(vendor_chunks.items())[:8]:
        primary = vchunks[0]

        # for catering vendors use per-plate price; for others scan chunks for package pricing
        if primary.price_veg_min:
            pricing = f"Rs {primary.price_veg_min}-{primary.price_veg_max}/plate (veg)"
            if primary.price_nonveg_min:
                pricing += f", Rs {primary.price_nonveg_min}-{primary.price_nonveg_max}/plate (non-veg)"
        else:
            pricing = None
            for c in vchunks:
                if any(kw in c.text.lower() for kw in PRICING_KEYWORDS):
                    pricing = c.text[:400]
                    break

        vendors.append({
            "name":       primary.vendor_name,
            "slug":       primary.vendor_slug,
            "city":       primary.city,
            "state":      primary.state,
            "categories": primary.categories[-2:] if primary.categories else [],
            "capacity":   f"{primary.capacity_min}-{primary.capacity_max} guests" if primary.capacity_max else None,
            "pricing":    pricing,
            "snippet":    primary.text[:200] if primary.text else "",
        })

    return json.dumps({"vendors": vendors, "total": len(vendors)}, ensure_ascii=False)


def get_vendor_details(vendor_slug: str) -> str:
    with _session() as session:
        vendor = session.query(Vendor).filter(Vendor.slug == vendor_slug).first()
        if not vendor:
            return json.dumps({"error": f"Vendor '{vendor_slug}' not found"})
        return json.dumps({
            "name":            vendor.name,
            "slug":            vendor.slug,
            "address":         vendor.address,
            "city":            vendor.city,
            "state":           vendor.state,
            "mobile":          vendor.mobile,
            "categories":      vendor.categories,
            "occasions":       vendor.occasions,
            "payment_methods": vendor.payment_methods,
            "capacity":        f"{vendor.capacity_min}-{vendor.capacity_max}" if vendor.capacity_max else "N/A",
            "price_veg":       f"Rs {vendor.price_veg_min}-{vendor.price_veg_max}/plate" if vendor.price_veg_min else "N/A",
            "price_nonveg":    f"Rs {vendor.price_nonveg_min}-{vendor.price_nonveg_max}/plate" if vendor.price_nonveg_min else "N/A",
            "description":     vendor.raw_description[:500] if vendor.raw_description else "",
        }, ensure_ascii=False)


def compare_vendors(vendor_slugs: list[str]) -> str:
    with _session() as session:
        vendors = session.query(Vendor).filter(Vendor.slug.in_(vendor_slugs)).all()
        comparison = [{
            "name":         v.name,
            "city":         v.city,
            "capacity":     f"{v.capacity_min}-{v.capacity_max}" if v.capacity_max else "N/A",
            "price_veg":    f"Rs {v.price_veg_min}-{v.price_veg_max}/plate" if v.price_veg_min else "N/A",
            "price_nonveg": f"Rs {v.price_nonveg_min}-{v.price_nonveg_max}/plate" if v.price_nonveg_min else "N/A",
            "occasions":    v.occasions or [],
            "payment":      v.payment_methods or [],
        } for v in vendors]
        return json.dumps({"comparison": comparison}, ensure_ascii=False)


def estimate_budget(guest_count: int, city: str, categories: list[str]) -> str:
    breakdown = []
    with _session() as session:
        for category in categories:
            results = (
                session.query(
                    func.min(Vendor.price_veg_min),
                    func.avg(Vendor.price_veg_min),
                    func.max(Vendor.price_veg_max),
                    func.count(Vendor.id),
                )
                .filter(
                    func.lower(Vendor.city) == city.lower(),
                    Vendor.price_veg_min.isnot(None),
                    func.lower(func.array_to_string(Vendor.categories, ",")).contains(category.lower()),
                )
                .first()
            )

            price_min, price_avg, price_max, count = results or (None, None, None, 0)

            if price_min is None:
                breakdown.append({
                    "category":     category,
                    "vendor_count": 0,
                    "note":         f"No price data found for {category} in {city}",
                })
            else:
                # for catering: multiply per-plate price by guest count
                is_catering = any(k in category.lower() for k in ("cater", "food", "chef"))
                if is_catering:
                    breakdown.append({
                        "category":        category,
                        "vendor_count":    count,
                        "per_plate_range": f"Rs {int(price_min)}-{int(price_max)}/plate",
                        "estimated_total": f"Rs {int(price_min * guest_count):,} - Rs {int(price_max * guest_count):,}",
                        "note":            f"Based on {guest_count} guests",
                    })
                else:
                    breakdown.append({
                        "category":        category,
                        "vendor_count":    count,
                        "price_range":     f"Rs {int(price_min):,} - Rs {int(price_max):,}",
                        "avg_price":       f"Rs {int(price_avg):,}",
                    })

    return json.dumps({
        "city":        city,
        "guest_count": guest_count,
        "breakdown":   breakdown,
    }, ensure_ascii=False)


def find_similar_vendors(vendor_slug: str, limit: int = 5) -> str:
    with _session() as session:
        ref = session.query(Vendor).filter(Vendor.slug == vendor_slug).first()
        if not ref:
            return json.dumps({"error": f"Vendor '{vendor_slug}' not found"})

        # build filter: same city, at least one overlapping category, exclude self
        q = session.query(Vendor).filter(
            Vendor.slug != vendor_slug,
            func.lower(Vendor.city) == (ref.city or "").lower(),
        )

        # filter by any overlapping category
        if ref.categories:
            q = q.filter(Vendor.categories.overlap(ref.categories))

        # narrow by price range ±50%
        if ref.price_veg_min:
            price_floor = int(ref.price_veg_min * 0.5)
            price_ceil  = int(ref.price_veg_min * 1.5)
            q = q.filter(
                and_(
                    Vendor.price_veg_min >= price_floor,
                    Vendor.price_veg_min <= price_ceil,
                )
            )

        similar = q.limit(limit).all()

        results = [{
            "name":         v.name,
            "slug":         v.slug,
            "city":         v.city,
            "categories":   v.categories or [],
            "capacity":     f"{v.capacity_min}-{v.capacity_max}" if v.capacity_max else "N/A",
            "price_veg":    f"Rs {v.price_veg_min}-{v.price_veg_max}/plate" if v.price_veg_min else "N/A",
        } for v in similar]

        return json.dumps({
            "reference_vendor": ref.name,
            "similar_vendors":  results,
            "total":            len(results),
        }, ensure_ascii=False)


# ── dispatcher ────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "search_vendors":     search_vendors,
    "get_vendor_details": get_vendor_details,
    "compare_vendors":    compare_vendors,
    "estimate_budget":    estimate_budget,
    "find_similar_vendors": find_similar_vendors,
}


def dispatch(tool_name: str, args: dict) -> str:
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return fn(**args)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})

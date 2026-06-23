import json
import logging
from sqlalchemy.orm import Session
from app.db.postgres import get_session
from app.models.vendor import Vendor
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)

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
                    "query": {"type": "string", "description": "Natural language search query"},
                    "city":  {"type": "string", "description": "Filter by city name"},
                    "state": {"type": "string", "description": "Filter by state name"},
                    "capacity_min": {"type": "integer", "description": "Minimum guest capacity needed"},
                    "price_veg_max": {"type": "integer", "description": "Max veg price per plate (₹)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vendor_details",
            "description": "Get full details of a specific vendor by their slug (folder name).",
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
]


# ── tool implementations ──────────────────────────────────────────────────────

def search_vendors(
    query: str,
    city: str | None = None,
    state: str | None = None,
    capacity_min: int | None = None,
    price_veg_max: int | None = None,
) -> str:
    filters = {}
    if city:           filters["city"] = city
    if state:          filters["state"] = state
    if capacity_min:   filters["capacity_min"] = capacity_min
    if price_veg_max:  filters["price_veg_max"] = price_veg_max

    chunks = retrieve(query, filters=filters, final_top_n=8)

    seen = set()
    vendors = []
    for c in chunks:
        if c.vendor_slug in seen:
            continue
        seen.add(c.vendor_slug)
        vendors.append({
            "name":            c.vendor_name,
            "slug":            c.vendor_slug,
            "city":            c.city,
            "state":           c.state,
            "categories":      c.categories[-2:] if c.categories else [],
            "capacity":        f"{c.capacity_min}-{c.capacity_max}" if c.capacity_max else "N/A",
            "price_veg":       f"₹{c.price_veg_min}-{c.price_veg_max}/plate" if c.price_veg_min else "N/A",
            "price_nonveg":    f"₹{c.price_nonveg_min}-{c.price_nonveg_max}/plate" if c.price_nonveg_min else "N/A",
            "snippet":         c.text[:200] if c.text else "",
        })

    return json.dumps({"vendors": vendors, "total": len(vendors)}, ensure_ascii=False)


def get_vendor_details(vendor_slug: str) -> str:
    session_gen = get_session()
    session: Session = next(session_gen)
    try:
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
            "price_veg":       f"₹{vendor.price_veg_min}-{vendor.price_veg_max}/plate" if vendor.price_veg_min else "N/A",
            "price_nonveg":    f"₹{vendor.price_nonveg_min}-{vendor.price_nonveg_max}/plate" if vendor.price_nonveg_min else "N/A",
            "description":     vendor.raw_description[:500] if vendor.raw_description else "",
        }, ensure_ascii=False)
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass


def compare_vendors(vendor_slugs: list[str]) -> str:
    session_gen = get_session()
    session: Session = next(session_gen)
    try:
        vendors = session.query(Vendor).filter(Vendor.slug.in_(vendor_slugs)).all()
        comparison = []
        for v in vendors:
            comparison.append({
                "name":       v.name,
                "city":       v.city,
                "capacity":   f"{v.capacity_min}-{v.capacity_max}" if v.capacity_max else "N/A",
                "price_veg":  f"₹{v.price_veg_min}-{v.price_veg_max}/plate" if v.price_veg_min else "N/A",
                "price_nonveg": f"₹{v.price_nonveg_min}-{v.price_nonveg_max}/plate" if v.price_nonveg_min else "N/A",
                "occasions":  v.occasions or [],
                "payment":    v.payment_methods or [],
            })
        return json.dumps({"comparison": comparison}, ensure_ascii=False)
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass


# ── dispatcher ────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "search_vendors":    search_vendors,
    "get_vendor_details": get_vendor_details,
    "compare_vendors":   compare_vendors,
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

import json
import re
import os
from pathlib import Path
from tqdm import tqdm
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

from app.db.postgres import get_engine
from app.models.vendor import Vendor
from sqlalchemy.orm import Session
from app.models.vendor_collection import VendorCollection
from app.rag.embedder import embed

DATA_DIR   = Path(os.getenv("DATA_DIR", "~/Documents/scrape_wire/data")).expanduser()
BATCH_SIZE = 32
CHECKPOINT = Path("embed_progress.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_price(text: str) -> tuple[int | None, int | None]:
    text = text.replace(",", "").replace("₹", "")
    nums = [int(x) for x in re.findall(r"\d+", text) if 500 <= int(x) <= 50_000]
    if not nums:
        return None, None
    return min(nums), max(nums)


def parse_capacity(text: str) -> tuple[int | None, int | None]:
    ranges = re.findall(r"(\d+)-(\d+)", text)
    if not ranges:
        return None, None
    all_nums = [int(n) for r in ranges for n in r]
    return min(all_nums), max(all_nums)


def parse_vendor(vendor_dir: Path) -> dict:
    vid   = vendor_dir.name
    cats  = load_json(vendor_dir / "category.json")
    cats  = cats if isinstance(cats, list) else []

    contact = load_json(vendor_dir / "contact.json")
    desc    = load_json(vendor_dir / "description.json")
    faqs    = load_json(vendor_dir / "faqs.json")
    images  = load_json(vendor_dir / "images.json")

    paragraphs = desc.get("description", []) if isinstance(desc, dict) else []
    faq_list   = faqs if isinstance(faqs, list) else []
    image_urls = images.get("links", []) if isinstance(images, dict) else []

    # geo
    coords = contact.get("coordinates", "") if isinstance(contact, dict) else ""
    lat = lng = None
    if coords:
        parts = coords.split(",")
        try:
            lat, lng = float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            pass

    # parse price + capacity from FAQs
    price_veg_min = price_veg_max = None
    price_nonveg_min = price_nonveg_max = None
    capacity_min = capacity_max = None
    occasions = []
    payment_methods = []

    for faq in faq_list:
        q = faq.get("question", "").lower()
        a = faq.get("answer", "")

        if "veg menu" in q and "non-veg" not in q:
            price_veg_min, price_veg_max = parse_price(a)
        if "non-veg menu" in q:
            price_nonveg_min, price_nonveg_max = parse_price(a)
        if "guest range" in q or "accommodate" in q:
            capacity_min, capacity_max = parse_capacity(a)
        if "occasion" in q:
            occasions = [o.strip() for o in re.split(r"[\n,]", a) if o.strip()]
        if "payment" in q:
            payment_methods = [p.strip() for p in re.split(r"[\n,]", a) if p.strip()]

    # vendor name = last item in categories usually
    name = cats[-1] if cats else vid

    return {
        "id":               vid,
        "name":             name,
        "address":          contact.get("address", "") if isinstance(contact, dict) else "",
        "city":             cats[-3] if len(cats) >= 3 else "",
        "state":            cats[-4] if len(cats) >= 4 else "",
        "lat":              lat,
        "lng":              lng,
        "mobile":           contact.get("mobile", "") if isinstance(contact, dict) else "",
        "categories":       cats,
        "occasions":        occasions,
        "payment_methods":  payment_methods,
        "capacity_min":     capacity_min,
        "capacity_max":     capacity_max,
        "price_veg_min":    price_veg_min,
        "price_veg_max":    price_veg_max,
        "price_nonveg_min": price_nonveg_min,
        "price_nonveg_max": price_nonveg_max,
        "image_urls":       image_urls,
        "raw_description":  " ".join(paragraphs),
        "raw_faqs":         faq_list,
        "_paragraphs":      paragraphs,   # used for chunking, not stored in DB
        "_faqs":            faq_list,
    }


def make_chunks(v: dict) -> list[dict]:
    chunks = []
    payload = {
        "vendor_id":   v["id"],
        "vendor_name": v["name"],
        "city":        v["city"],
        "state":       v["state"],
        "categories":  v["categories"],
        "capacity_min":     v["capacity_min"],
        "capacity_max":     v["capacity_max"],
        "price_veg_min":    v["price_veg_min"],
        "price_veg_max":    v["price_veg_max"],
        "price_nonveg_min": v["price_nonveg_min"],
        "price_nonveg_max": v["price_nonveg_max"],
    }

    # summary chunk
    summary = " | ".join(filter(None, [
        v["name"],
        ", ".join(v["categories"][:4]),
        f"{v['city']}, {v['state']}",
        f"Capacity {v['capacity_min']}-{v['capacity_max']}" if v["capacity_max"] else "",
        f"Veg ₹{v['price_veg_min']}-{v['price_veg_max']}/head" if v["price_veg_min"] else "",
        f"Non-veg ₹{v['price_nonveg_min']}-{v['price_nonveg_max']}/head" if v["price_nonveg_min"] else "",
        f"Occasions: {', '.join(v['occasions'])}" if v["occasions"] else "",
        f"Phone: {v['mobile']}" if v["mobile"] else "",
    ]))
    chunks.append({"id": f"{v['id']}_summary", "text": summary,
                   "payload": {**payload, "chunk_type": "summary"}})

    # description chunks
    for i, para in enumerate(v["_paragraphs"]):
        para = re.sub(r"\*+", "", para).strip()
        if len(para) < 50:
            continue
        chunks.append({"id": f"{v['id']}_desc_{i}", "text": para[:900],
                       "payload": {**payload, "chunk_type": "description"}})

    # faq chunks
    seen = set()
    for i, faq in enumerate(v["_faqs"]):
        q = faq.get("question", "").strip()
        a = re.sub(r"\s+", " ", faq.get("answer", "")).strip()
        if not q or not a or len(a) < 10 or q in seen:
            continue
        seen.add(q)
        chunks.append({"id": f"{v['id']}_faq_{i}", "text": f"Q: {q}\nA: {a[:600]}",
                       "payload": {**payload, "chunk_type": "faq"}})

    return chunks


def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT.write_text(json.dumps(list(done)))


# ── main ─────────────────────────────────────────────────────────────────────

DB_BATCH_SIZE = 500   # vendors per postgres batch

def run():
    print("Setting up Qdrant collection...")
    VendorCollection.create()

    vendor_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(vendor_dirs)} vendors.")

    done = load_checkpoint()
    pending_chunks = []
    db_batch = []

    print("Parsing vendors...")
    for vendor_dir in tqdm(vendor_dirs):
        v = parse_vendor(vendor_dir)
        db_data = {k: val for k, val in v.items() if not k.startswith("_")}
        db_batch.append(db_data)

        # flush postgres batch
        if len(db_batch) >= DB_BATCH_SIZE:
            with Session(get_engine()) as session:
                Vendor.bulk_insert(session, db_batch)
                session.commit()
            db_batch = []

        for chunk in make_chunks(v):
            if chunk["id"] not in done:
                pending_chunks.append(chunk)

    # flush remaining postgres batch
    if db_batch:
        with Session(get_engine()) as session:
            Vendor.bulk_insert(session, db_batch)
            session.commit()

    print(f"{len(pending_chunks)} chunks to embed.")

    # embed in batches
    for i in tqdm(range(0, len(pending_chunks), BATCH_SIZE), desc="Embedding"):
        batch = pending_chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        points = [
            PointStruct(id=c["id"], vector=v, payload=c["payload"])
            for c, v in zip(batch, vectors)
        ]
        VendorCollection.upsert(points)

        done.update(c["id"] for c in batch)
        if i % (BATCH_SIZE * 100) == 0:
            save_checkpoint(done)

    save_checkpoint(done)
    print("Ingestion complete.")


if __name__ == "__main__":
    run()

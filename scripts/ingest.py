import json
import re
import os
import time
import uuid
import itertools
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ulid import ULID
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
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "512"))
PARSE_WORKERS = int(os.getenv("INGEST_PARSE_WORKERS", "16"))
PARSE_AHEAD   = int(os.getenv("INGEST_PARSE_AHEAD", "2000"))  # vendors prefetched
CHECKPOINT = Path("embed_progress.json")
CHECKPOINT_EVERY = 10_000  # chunks


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
        "id":               str(ULID()),
        "slug":             vid,
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
        "vendor_id":   v["id"],    # ULID
        "vendor_slug": v["slug"],  # original folder name
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
    chunks.append({"id": f"{v['slug']}_summary", "text": summary,
                   "payload": {**payload, "chunk_type": "summary"}})

    # description chunks
    for i, para in enumerate(v["_paragraphs"]):
        para = re.sub(r"\*+", "", para).strip()
        if len(para) < 50:
            continue
        chunks.append({"id": f"{v['slug']}_desc_{i}", "text": para[:900],
                       "payload": {**payload, "chunk_type": "description"}})

    # faq chunks
    seen = set()
    for i, faq in enumerate(v["_faqs"]):
        q = faq.get("question", "").strip()
        a = re.sub(r"\s+", " ", faq.get("answer", "")).strip()
        if not q or not a or len(a) < 10 or q in seen:
            continue
        seen.add(q)
        chunks.append({"id": f"{v['slug']}_faq_{i}", "text": f"Q: {q}\nA: {a[:600]}",
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

def parse_one(vendor_dir: Path) -> tuple[dict, list[dict]]:
    """Parse a vendor dir into (db_row, chunks). Runs in worker threads — the
    file reads here release the GIL, so this overlaps with GPU embedding."""
    v = parse_vendor(vendor_dir)
    db_data = {k: val for k, val in v.items() if not k.startswith("_")}
    return db_data, make_chunks(v)


def prefetch(fn, items, workers: int, ahead: int):
    """Yield fn(item) in order, keeping up to `ahead` calls in flight across a
    thread pool. Bounded memory; keeps the GPU fed without reading everything."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(items)
        q: deque = deque()
        for x in itertools.islice(it, ahead):
            q.append(ex.submit(fn, x))
        for x in it:
            yield q.popleft().result()
            q.append(ex.submit(fn, x))
        while q:
            yield q.popleft().result()


def make_points(batch: list[dict]) -> list[PointStruct]:
    texts = [c["text"] for c in batch]
    vectors = embed(texts)
    return [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["id"])),
            vector=v,
            payload={**c["payload"], "chunk_id": c["id"], "text": c["text"]},
        )
        for c, v in zip(batch, vectors)
    ]


def run(limit: int | None = None):
    print("Setting up Qdrant collection...")
    VendorCollection.create()

    vendor_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if limit:
        vendor_dirs = vendor_dirs[:limit]
    print(f"Found {len(vendor_dirs)} vendors.")

    done = load_checkpoint()
    chunk_buffer: list[dict] = []
    db_batch: list[dict] = []
    chunks_done = 0

    # per-stage wall-clock, so the bottleneck is visible (parse vs GPU vs network)
    t_parse_wait = t_gpu = t_upsert_wait = t_db = 0.0

    # track in-flight upsert alongside its batch so we only mark done after confirm
    inflight_future = None
    inflight_batch: list[dict] = []

    def confirm_inflight():
        nonlocal inflight_future, inflight_batch, chunks_done, t_upsert_wait
        if inflight_future is None:
            return
        t = time.perf_counter()
        inflight_future.result()           # raises if upsert failed after retries
        t_upsert_wait += time.perf_counter() - t
        done.update(c["id"] for c in inflight_batch)
        chunks_done += len(inflight_batch)
        if chunks_done % CHECKPOINT_EVERY < len(inflight_batch):
            save_checkpoint(done)
        inflight_future = None
        inflight_batch = []

    def submit_batch(batch: list[dict]):
        nonlocal inflight_future, inflight_batch, t_gpu
        confirm_inflight()                 # wait for previous before GPU starts next
        t = time.perf_counter()
        points = make_points(batch)        # GPU forward pass
        t_gpu += time.perf_counter() - t
        inflight_future = pool.submit(VendorCollection.upsert, points)
        inflight_batch = batch

    # one long-lived connection beats reconnecting to Neon every batch (NullPool)
    db_session = Session(get_engine())

    def flush_db():
        nonlocal db_batch, t_db
        if not db_batch:
            return
        t = time.perf_counter()
        Vendor.bulk_insert(db_session, db_batch)
        db_session.commit()
        t_db += time.perf_counter() - t
        db_batch = []

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            # parse the next PARSE_AHEAD vendors on worker threads while the GPU
            # works the current batch — file I/O releases the GIL.
            parsed = prefetch(parse_one, vendor_dirs, PARSE_WORKERS, PARSE_AHEAD)

            with tqdm(total=len(vendor_dirs), desc="Ingesting") as pbar:
                while True:
                    t = time.perf_counter()
                    item = next(parsed, None)          # blocks only if GPU outran parsing
                    t_parse_wait += time.perf_counter() - t
                    if item is None:
                        break
                    db_data, chunks = item
                    db_batch.append(db_data)
                    if len(db_batch) >= DB_BATCH_SIZE:
                        flush_db()

                    for chunk in chunks:
                        if chunk["id"] not in done:
                            chunk_buffer.append(chunk)

                    while len(chunk_buffer) >= BATCH_SIZE:
                        batch = chunk_buffer[:BATCH_SIZE]
                        chunk_buffer = chunk_buffer[BATCH_SIZE:]
                        submit_batch(batch)
                    pbar.update(1)

            # flush remaining chunks
            if chunk_buffer:
                submit_batch(chunk_buffer)

            confirm_inflight()  # wait for last upsert and mark it done
            flush_db()
    finally:
        db_session.close()

    save_checkpoint(done)
    print(f"Ingestion complete. {chunks_done} chunks embedded.")
    print(f"Stage time (s): parse_wait={t_parse_wait:.0f} "
          f"gpu={t_gpu:.0f} upsert_wait={t_upsert_wait:.0f} db={t_db:.0f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N vendors (for testing)")
    args = parser.parse_args()
    run(limit=args.limit)

"""embed_search.py — One-time script to build the AI search index.

For each comic, an LLM generates 5 vivid search phrases that capture the
joke/theme. Those phrases are then embedded with text-embedding-3-small via
OpenAI and stored in R2 as search_index.json.

The app loads this file at startup and uses cosine similarity to power the
/api/suggest endpoint — zero per-search API cost, ~instant responses.

Usage:
    python embed_search.py              # process all comics
    python embed_search.py --resume     # skip comics already in search_index.json
    python embed_search.py --dry-run    # show phrases for first 5 comics, no upload

Cost estimate (one-time, via OpenAI):
    - Phrase gen (gpt-4.1-mini): ~675 × ~200 tokens ≈ $0.01
    - Embeddings (text-embedding-3-small): ~675 × 5 phrases × ~8 tokens ≈ $0.001
    Total: well under $0.05
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "work-chronicles-storage")
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]

POSTS_KEY = "posts.json"
INDEX_KEY = "search_index.json"

PHRASE_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

EMBED_BATCH_SIZE = 100
PHRASE_WORKERS = None  # None = unlimited (one thread per comic)
PHRASE_TIMEOUT = 45  # seconds per LLM request before giving up

# ── Clients ───────────────────────────────────────────────────────────────────

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(
        signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}
    ),
    region_name="auto",
)


def _make_openai():
    return OpenAI(api_key=OPENAI_API_KEY)


# ── R2 helpers ────────────────────────────────────────────────────────────────


def load_posts() -> list[dict]:
    log.info("Loading posts.json from R2...")
    res = s3.get_object(Bucket=R2_BUCKET, Key=POSTS_KEY)
    data = json.loads(res["Body"].read())
    posts = list(data.get("posts", {}).values())
    posts.sort(key=lambda p: p.get("post_date", ""), reverse=True)
    log.info("Loaded %d posts.", len(posts))
    return posts


def load_existing_index() -> dict:
    """Load existing search_index.json from R2, or return empty structure."""
    try:
        res = s3.get_object(Bucket=R2_BUCKET, Key=INDEX_KEY)
        data = json.loads(res["Body"].read())
        log.info(
            "Found existing search index with %d entries.", len(data.get("entries", []))
        )
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            log.info("No existing search index found — starting fresh.")
            return {"entries": []}
        raise


def save_index(index: dict) -> None:
    body = json.dumps(index, separators=(",", ":")).encode()
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=INDEX_KEY,
        Body=body,
        ContentType="application/json",
    )
    log.info(
        "Saved search index (%d entries, %.1f KB) to R2.",
        len(index["entries"]),
        len(body) / 1024,
    )


# ── Phrase generation ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are building a search index for a workplace comic strip gallery called Work Chronicles.
Given a comic title, output exactly 5 short search phrases (2-6 words each) that capture
what someone might type to find this comic.

Think about:
- The core joke or workplace frustration depicted
- The type of colleague, manager, or situation shown
- Synonyms and alternate phrasings a user might type
- Emotional tone (e.g. "passive aggressive email", "boss takes credit", "useless meeting")

Return ONLY a JSON array of 5 strings. No explanation, no markdown, no extra keys.
Example output: ["boss micromanaging", "manager breathing down neck", "no autonomy", "constant check-ins", "trust issues at work"]
"""


def generate_phrases(title: str, retries: int = 3) -> list[str]:
    """Ask the LLM to generate 5 search phrases for a comic title."""
    client = _make_openai()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=PHRASE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Comic title: "{title}"'},
                ],
                temperature=0.7,
                max_tokens=150,
                timeout=PHRASE_TIMEOUT,
            )
            content = resp.choices[0].message.content or ""
            raw = content.strip()

            if not raw:
                time.sleep(1.5**attempt)
                continue

            # Strip markdown code fences if the model wraps output
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            if isinstance(parsed, list):
                phrases = parsed
            else:
                phrases = next((v for v in parsed.values() if isinstance(v, list)), [])

            phrases = [str(p).strip() for p in phrases if str(p).strip()][:5]
            if phrases:
                return phrases

        except Exception as e:
            wait = 1.5**attempt
            log.warning(
                "Phrase gen failed (attempt %d/%d) for '%s': %s",
                attempt + 1,
                retries,
                title,
                e,
            )
            time.sleep(wait)

    log.warning("All retries exhausted for '%s' — using title as fallback.", title)
    return [title.lower()]


# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using text-embedding-3-small via OpenAI."""
    client = _make_openai()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_vectors.extend(vectors)
        log.info(
            "  Embedded %d/%d phrases...",
            min(i + EMBED_BATCH_SIZE, len(texts)),
            len(texts),
        )
    return all_vectors


# ── Main ──────────────────────────────────────────────────────────────────────


def build_index(resume: bool = False, dry_run: bool = False) -> None:
    posts = load_posts()
    existing_index = load_existing_index() if resume else {"entries": []}

    indexed_ids = {e["id"] for e in existing_index.get("entries", [])}
    to_process = [p for p in posts if p["id"] not in indexed_ids]

    if not to_process:
        log.info("All %d posts already indexed. Nothing to do.", len(posts))
        return

    log.info(
        "Processing %d comics (%d already indexed).",
        len(to_process),
        len(indexed_ids),
    )

    if dry_run:
        sample = to_process[:5]
        log.info("DRY RUN — showing phrases for first %d comics:", len(sample))
        for post in sample:
            phrases = generate_phrases(post["title"])
            print(f"\n  [{post['id']}] {post['title']}")
            for p in phrases:
                print(f"    → {p}")
        return

    # ── Phase 1: generate phrases in parallel ────────────────────────────────
    log.info(
        "Phase 1/2 — generating search phrases with %s (%s workers)...",
        PHRASE_MODEL,
        PHRASE_WORKERS if PHRASE_WORKERS else "unlimited",
    )
    posts_with_phrases: list[tuple[dict, list[str]]] = [None] * len(to_process)  # type: ignore[list-item]
    total = len(to_process)
    done_count = 0
    phase1_start = time.time()

    with ThreadPoolExecutor(max_workers=PHRASE_WORKERS) as pool:
        future_to_idx = {
            pool.submit(generate_phrases, post["title"]): i
            for i, post in enumerate(to_process)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                phrases = future.result()
            except Exception as e:
                log.error(
                    "Unexpected error for comic %s: %s", to_process[idx]["title"], e
                )
                phrases = [to_process[idx]["title"].lower()]
            posts_with_phrases[idx] = (to_process[idx], phrases)
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                elapsed = time.time() - phase1_start
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = total - done_count
                eta_s = remaining / rate if rate > 0 else 0
                log.info(
                    "  Phrases: %d/%d  (%.1f/s, ETA ~%.0fs)",
                    done_count,
                    total,
                    rate,
                    eta_s,
                )

    # ── Phase 2: embed all phrases in batches ────────────────────────────────
    log.info("Phase 2/2 — embedding phrases with %s...", EMBED_MODEL)

    all_phrases_flat: list[str] = []
    phrase_map: list[tuple[dict, int, int]] = []  # (post, start_idx, count)

    for post, phrases in posts_with_phrases:
        phrase_map.append((post, len(all_phrases_flat), len(phrases)))
        all_phrases_flat.extend(phrases)

    log.info("  Total phrases to embed: %d", len(all_phrases_flat))
    all_vectors = embed_texts(all_phrases_flat)

    # ── Assemble index entries ───────────────────────────────────────────────
    new_entries: list[dict] = []
    for post, start_idx, count in phrase_map:
        phrase_vectors = all_vectors[start_idx : start_idx + count]
        phrases = all_phrases_flat[start_idx : start_idx + count]

        mean_vec = np.mean(phrase_vectors, axis=0).tolist()

        new_entries.append(
            {
                "id": post["id"],
                "title": post["title"],
                "category": post.get("category", ""),
                "ext": post.get("ext", "jpg"),
                "post_date": post.get("post_date", ""),
                "phrases": phrases,
                "vector": mean_vec,
            }
        )

    merged_entries = list(existing_index.get("entries", [])) + new_entries
    final_index = {
        "entries": merged_entries,
        "model": EMBED_MODEL,
        "dim": len(merged_entries[0]["vector"]) if merged_entries else 0,
    }
    save_index(final_index)
    log.info("Done. Index has %d entries total.", len(merged_entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build AI search index for Work Chronicles Gallery."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip comics already present in search_index.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generated phrases for first 5 comics without uploading",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY is not set in your environment or .env file.")
        sys.exit(1)

    # Quick sanity check: verify the model returns parseable content
    log.info("Testing model connection...")
    try:
        test_phrases = generate_phrases("test comic")
        log.info("Model returned: %s", test_phrases)
        if not test_phrases or test_phrases == ["test comic"]:
            log.error(
                "Model returned no usable phrases — check if %s is available.",
                PHRASE_MODEL,
            )
            sys.exit(1)
    except Exception as e:
        log.error("Model test failed: %s", e)
        sys.exit(1)

    build_index(resume=args.resume, dry_run=args.dry_run)

import os
import json
import logging
import time
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import requests

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — verbose, structured output for easy debugging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_API = "https://www.workchronicles.com/api/v1/archive"
METADATA_FILE = "posts.json"  # must match POSTS_KEY in app.py
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 12
API_LIMIT = 50
# Prefix stripped from post titles before storing; extracted as a named constant
# so a future API change is a one-line update.
TITLE_PREFIX = "(comic) "

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "work-chronicles-storage")
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(
        signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}
    ),
    region_name="auto",
)

# Module-level session — shared across all threads; avoids per-post TCP/TLS
# handshakes and improves throughput significantly under the thread pool.
_session = requests.Session()
_session.headers.update(HEADERS)


def load_metadata():
    """Load posts.json from R2. Returns empty metadata on NoSuchKey.
    Re-raises all other ClientErrors so misconfiguration is not silently ignored.
    """
    log.debug("Loading metadata from R2 (key=%s, bucket=%s)", METADATA_FILE, R2_BUCKET)
    try:
        res = s3.get_object(Bucket=R2_BUCKET, Key=METADATA_FILE)
        data = json.loads(res["Body"].read())
        log.info("Loaded metadata: %d known posts", len(data.get("posts", {})))
        return data
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchKey":
            log.info("No existing metadata found in R2 — starting fresh.")
            return {"last_sync": None, "posts": {}}
        log.error("R2 ClientError loading metadata (%s): %s", code, e)
        raise
    except json.JSONDecodeError as e:
        log.error("posts.json is corrupted (invalid JSON): %s", e)
        raise


def save_metadata(data):
    log.debug("Saving metadata to R2 (%d posts)...", len(data.get("posts", {})))
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=METADATA_FILE,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )
    log.info("Metadata saved to R2 successfully.")


def safe_get(url, retries=3):
    """GET *url* with retries. Returns a Response or None on all failures."""
    log.debug("GET %s (retries=%d)", url, retries)
    for attempt in range(retries):
        try:
            res = _session.get(url, timeout=15)
            if res.status_code == 200:
                log.debug("GET %s -> 200 OK (attempt %d)", url, attempt + 1)
                return res
            if res.status_code == 429:
                wait = 2**attempt
                log.warning(
                    "GET %s -> 429 Rate Limited (attempt %d/%d), waiting %ds",
                    url,
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            log.warning("GET %s -> %d (non-retryable)", url, res.status_code)
            return None
        except Exception as e:
            if attempt < retries - 1:
                log.warning(
                    "GET %s failed (attempt %d/%d): %s — retrying in 1s",
                    url,
                    attempt + 1,
                    retries,
                    e,
                )
                time.sleep(1)
                continue
            log.error("GET %s failed after %d attempts: %s", url, retries, e)
            return None
    # All retries exhausted (only reachable if every attempt returned 429).
    log.error("GET %s exhausted all %d retries (all rate-limited).", url, retries)
    return None


def download_and_upload(post):
    post_id = post["id"]
    img_url = post.get("cover_image")
    if not img_url:
        log.debug("Post %d has no cover_image — skipping.", post_id)
        return None, post_id, None

    log.debug("Downloading post %d: %s", post_id, img_url)
    try:
        res = safe_get(img_url)
        if not res:
            log.warning("Post %d: download returned no response.", post_id)
            return None, post_id, None

        content_type = (
            res.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        )
        ext = content_type.split("/")[-1] if "/" in content_type else "jpg"
        key = f"{post_id}.{ext}"

        log.debug(
            "Uploading post %d as %s (content_type=%s)", post_id, key, content_type
        )
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=res.content,
            ContentType=content_type,
            # Immutable cache — images never change once uploaded.
            CacheControl="max-age=31536000, immutable",
        )
        log.debug("Post %d uploaded successfully as %s", post_id, key)
        return key, post_id, ext
    except Exception as e:
        log.error(
            "Post %d: unexpected error during download/upload: %s",
            post_id,
            e,
            exc_info=True,
        )
        return None, post_id, None


def _fetch_pages(stop_condition=None):
    """Generator that yields pages (lists) from the archive API.

    The API returns posts newest-first. *stop_condition*, if provided, is a
    callable that receives each post dict and returns True when pagination
    should stop (e.g., when a known post ID is encountered).

    NOTE: This relies on the API returning posts in strictly descending
    chronological order. If the API ever returns posts out-of-order,
    incremental sync may silently miss new posts that appear after a known ID.

    NOTE: offset advances by len(data) — the raw API page count — not by
    len(filtered). This is intentional: the API uses absolute offsets so we
    must advance by the number of records the API actually returned.
    """
    offset = 0
    page_num = 0
    while True:
        url = f"{BASE_API}?offset={offset}&limit={API_LIMIT}"
        log.debug("Fetching page %d (offset=%d): %s", page_num, offset, url)
        res = safe_get(url)
        if not res:
            log.error(
                "Failed to fetch page %d (offset=%d) — stopping pagination.",
                page_num,
                offset,
            )
            break
        try:
            data = res.json()
        except Exception as e:
            log.error("Page %d: failed to parse JSON response: %s", page_num, e)
            break
        if not isinstance(data, list) or len(data) == 0:
            log.info("Page %d returned empty — pagination complete.", page_num)
            break

        log.debug("Page %d: received %d posts", page_num, len(data))

        if stop_condition:
            filtered = []
            hit_known = False
            for post in data:
                if stop_condition(post):
                    log.debug(
                        "Page %d: stop_condition hit at post id=%d after %d new posts.",
                        page_num,
                        post["id"],
                        len(filtered),
                    )
                    hit_known = True
                    break
                filtered.append(post)
            if filtered:
                yield filtered
            if hit_known:
                return
        else:
            yield data

        offset += len(data)
        page_num += 1
        time.sleep(0.3)


def fetch_all_posts():
    """Return every post from the API."""
    all_posts = []
    for page in _fetch_pages():
        all_posts.extend(page)
    return all_posts


def process_posts(posts, existing_metadata=None):
    post_map = {p["id"]: p for p in posts}
    results = {}
    uploaded = 0
    reused = 0
    failed = 0

    log.info("Processing %d posts with %d workers...", len(posts), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(download_and_upload, post): post["id"] for post in posts
        }

        done_count = 0
        total = len(future_to_id)
        for future in as_completed(future_to_id):
            key, post_id, ext = future.result()
            done_count += 1
            if key:
                post = post_map[post_id]
                title = post["title"].removeprefix(TITLE_PREFIX)
                results[str(post_id)] = {
                    "id": post["id"],
                    "title": title,
                    "slug": post.get("slug", ""),
                    "post_date": post.get("post_date", ""),
                    "canonical_url": post.get("canonical_url", ""),
                    "ext": ext,
                }
                uploaded += 1
                log.info("[%d/%d] Uploaded: %s", done_count, total, title)
            else:
                if existing_metadata and str(post_id) in existing_metadata.get(
                    "posts", {}
                ):
                    results[str(post_id)] = existing_metadata["posts"][str(post_id)]
                    reused += 1
                    log.info(
                        "[%d/%d] Reused existing: %s",
                        done_count,
                        total,
                        results[str(post_id)]["title"],
                    )
                else:
                    failed += 1
                    log.warning(
                        "[%d/%d] Post %d failed to download and has no existing metadata — skipped.",
                        done_count,
                        total,
                        post_id,
                    )

    log.info(
        "process_posts complete: %d uploaded, %d reused, %d failed.",
        uploaded,
        reused,
        failed,
    )
    return results


def full_sync():
    log.info("=== FULL SYNC ===")
    log.info("Fetching all posts from API...")
    all_posts = fetch_all_posts()
    log.info("Found %d posts total", len(all_posts))

    metadata = {"last_sync": datetime.now(timezone.utc).isoformat(), "posts": {}}
    results = process_posts(all_posts)
    metadata["posts"] = results

    save_metadata(metadata)
    log.info("Full sync done: %d images in R2", len(results))
    return len(results), 0


def incremental_sync():
    log.info("=== INCREMENTAL SYNC ===")
    metadata = load_metadata()

    # Safely convert keys to ints; skip malformed keys with a warning.
    known_ids = set()
    for k in metadata["posts"].keys():
        try:
            known_ids.add(int(k))
        except ValueError:
            log.warning("Skipping non-integer post key in metadata: %r", k)

    log.info("Known posts: %d", len(known_ids))

    new_posts = []
    for page in _fetch_pages(stop_condition=lambda p: p["id"] in known_ids):
        new_posts.extend(page)

    if not new_posts:
        log.info("No new posts found — already up to date.")
        return 0, len(known_ids)

    log.info("Found %d new posts to process", len(new_posts))
    results = process_posts(new_posts, metadata)

    metadata["posts"].update(results)
    metadata["last_sync"] = datetime.now(timezone.utc).isoformat()
    save_metadata(metadata)
    log.info(
        "Incremental sync done: %d new posts processed, %d total known.",
        len(results),
        len(known_ids) + len(results),
    )
    return len(results), len(known_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work Chronicles image downloader")
    parser.add_argument("--full", action="store_true", help="Force full sync")
    parser.add_argument(
        "--incremental", action="store_true", help="Force incremental sync"
    )
    args = parser.parse_args()

    if args.full:
        full_sync()
    elif args.incremental:
        incremental_sync()
    else:
        # Always try incremental first; falls back to full if no metadata in R2
        metadata = load_metadata()
        if metadata["posts"]:
            incremental_sync()
        else:
            full_sync()

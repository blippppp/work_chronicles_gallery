import os
import json
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

BASE_API = "https://www.workchronicles.com/api/v1/archive"
METADATA_FILE = "posts.json"
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
    try:
        res = s3.get_object(Bucket=R2_BUCKET, Key=METADATA_FILE)
        return json.loads(res["Body"].read())
    except ClientError:
        return {"last_sync": None, "posts": {}}


def save_metadata(data):
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=METADATA_FILE,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )


def safe_get(url, retries=3):
    """GET *url* with retries. Returns a Response or None on all failures."""
    for attempt in range(retries):
        try:
            res = _session.get(url, timeout=15)
            if res.status_code == 200:
                return res
            if res.status_code == 429:
                time.sleep(2**attempt)
                continue
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            print(f"  safe_get {url} failed after {retries} attempts: {e}")
            return None
    return None


def download_and_upload(post):
    post_id = post["id"]
    img_url = post.get("cover_image")
    if not img_url:
        return None, post_id, None

    try:
        res = safe_get(img_url)
        if not res:
            return None, post_id, None

        # Use actual Content-Type from response instead of assuming JPEG.
        content_type = (
            res.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        )
        ext = content_type.split("/")[-1] if "/" in content_type else "jpg"
        key = f"{post_id}.{ext}"

        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=res.content,
            ContentType=content_type,
            # Immutable cache — images never change once uploaded.
            CacheControl="max-age=31536000, immutable",
        )
        return key, post_id, ext
    except Exception as e:
        print(f"  Error on post {post_id}: {e}")
        return None, post_id, None


def _fetch_pages(stop_condition=None):
    """Generator that yields pages (lists) from the archive API.

    The API returns posts newest-first. *stop_condition*, if provided, is a
    callable that receives each post dict and returns True when pagination
    should stop (e.g., when a known post ID is encountered).

    NOTE: This relies on the API returning posts in strictly descending
    chronological order. If the API ever returns posts out-of-order,
    incremental sync may silently miss new posts that appear after a known ID.
    """
    offset = 0
    while True:
        url = f"{BASE_API}?offset={offset}&limit={API_LIMIT}"
        res = safe_get(url)
        if not res:
            break
        try:
            data = res.json()
        except Exception:
            break
        if not isinstance(data, list) or len(data) == 0:
            break

        if stop_condition:
            filtered = []
            hit_known = False
            for post in data:
                if stop_condition(post):
                    hit_known = True
                    break
                filtered.append(post)
            yield filtered
            if hit_known:
                return
        else:
            yield data

        offset += len(data)
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
                results[str(post_id)] = {
                    "id": post["id"],
                    "title": post["title"].replace(TITLE_PREFIX, ""),
                    "slug": post.get("slug", ""),
                    "post_date": post.get("post_date", ""),
                    "canonical_url": post.get("canonical_url", ""),
                    "ext": ext,
                }
                print(f"  [{done_count}/{total}] {results[str(post_id)]['title']}")
            else:
                if existing_metadata and str(post_id) in existing_metadata.get(
                    "posts", {}
                ):
                    results[str(post_id)] = existing_metadata["posts"][str(post_id)]
                    print(
                        f"  [{done_count}/{total}] (skipped/reused) {results[str(post_id)]['title']}"
                    )

    return results


def full_sync():
    print("=== FULL SYNC ===")
    print("Fetching all posts...")
    all_posts = fetch_all_posts()
    print(f"Found {len(all_posts)} posts total")

    metadata = {"last_sync": datetime.now(timezone.utc).isoformat(), "posts": {}}
    results = process_posts(all_posts)
    metadata["posts"] = results

    save_metadata(metadata)
    print(f"\nDone: {len(results)} images downloaded to R2")
    return len(results), 0


def incremental_sync():
    print("=== INCREMENTAL SYNC ===")
    metadata = load_metadata()
    known_ids = set(int(k) for k in metadata["posts"].keys())
    print(f"Known posts: {len(known_ids)}")

    new_posts = []
    for page in _fetch_pages(stop_condition=lambda p: p["id"] in known_ids):
        new_posts.extend(page)

    if not new_posts:
        print("No new posts found.")
        return 0, len(known_ids)

    print(f"Found {len(new_posts)} new posts")
    results = process_posts(new_posts, metadata)

    metadata["posts"].update(results)
    metadata["last_sync"] = datetime.now(timezone.utc).isoformat()
    save_metadata(metadata)
    print(f"\nDone: {len(new_posts)} new images downloaded to R2")
    return len(new_posts), len(known_ids)


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

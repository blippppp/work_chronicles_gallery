import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from categories import CATEGORIES, DEFAULT_CATEGORY

load_dotenv()

app = Flask(__name__)

R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "work-chronicles-storage")
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
SYNC_TOKEN = os.environ.get("SYNC_TOKEN", "")
BATCH_SIZE = 40
POSTS_KEY = "posts.json"  # must match METADATA_FILE in image_downloader.py

# Warn loudly on startup if critical config is missing or insecure.
if not R2_PUBLIC_URL:
    app.logger.warning("R2_PUBLIC_URL is not set — all image URLs will be broken.")
if not SYNC_TOKEN:
    app.logger.warning(
        "SYNC_TOKEN is not set — /api/sync is unauthenticated. "
        "Set SYNC_TOKEN in your environment to protect this endpoint."
    )

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

# In-memory cache protected by a lock.
# Under Gunicorn multi-worker each worker has its own copy; sync updates it
# within the worker that handled the request. Use a shared data store
# (Redis / SQLite) if strict cross-worker consistency is required.
_cache_lock = threading.Lock()
_ALL_POSTS: list = []


def load_posts() -> list:
    """Fetch posts.json from R2 and return posts_list.

    Raises ClientError for unexpected errors (non-NoSuchKey) so callers
    can decide how to handle them.
    Raises json.JSONDecodeError if R2 returns corrupted data.
    """
    try:
        res = s3.get_object(Bucket=R2_BUCKET, Key=POSTS_KEY)
        raw = res["Body"].read()
        data = json.loads(raw)
        posts = list(data.get("posts", {}).values())
        posts.sort(key=lambda p: p.get("post_date", ""), reverse=True)
        return posts
    except json.JSONDecodeError as e:
        app.logger.error("posts.json is corrupted (invalid JSON): %s", e)
        raise
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchKey":
            # Not yet synced — expected on first run.
            return []
        app.logger.error("R2 load_posts error (%s): %s", code, e)
        raise


def _refresh_cache() -> None:
    """Re-read R2 and update the in-memory cache under the lock."""
    posts = load_posts()
    with _cache_lock:
        global _ALL_POSTS
        _ALL_POSTS = posts


# Populate cache at startup (errors here are intentionally fatal).
_refresh_cache()


@app.route("/")
def index():
    with _cache_lock:
        first_batch = _ALL_POSTS[:BATCH_SIZE]
        total = len(_ALL_POSTS)
    return render_template(
        "index.html",
        posts=first_batch,
        r2_base_url=R2_PUBLIC_URL,
        total=total,
    )


@app.route("/api/images")
def get_images_api():
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "offset must be an integer"}), 400
    with _cache_lock:
        batch = _ALL_POSTS[offset : offset + BATCH_SIZE]
    return jsonify(batch)


@app.route("/api/search")
def search_images():
    query = request.args.get("q", "").lower().strip()
    if not query:
        return jsonify([])
    with _cache_lock:
        results = [p for p in _ALL_POSTS if query in p.get("title", "").lower()]
    return jsonify(results[:200])


@app.route("/api/categories")
def get_categories():
    """Return category metadata with counts."""
    with _cache_lock:
        counts: dict[str, int] = {}
        for p in _ALL_POSTS:
            cat = p.get("category", DEFAULT_CATEGORY)
            counts[cat] = counts.get(cat, 0) + 1

    result = []
    for cat_name, cat_info in CATEGORIES.items():
        result.append(
            {
                "name": cat_name,
                "icon": cat_info["icon"],
                "description": cat_info["description"],
                "count": counts.get(cat_name, 0),
            }
        )

    ann_count = counts.get("Announcements", 0)
    if ann_count:
        result.append(
            {
                "name": "Announcements",
                "icon": "📢",
                "description": "Calendar releases, milestones, and meta content",
                "count": ann_count,
            }
        )

    result.sort(key=lambda c: c["count"], reverse=True)
    return jsonify(result)


@app.route("/api/category/<name>")
def get_category(name: str):
    """Return all posts in a category."""
    with _cache_lock:
        results = [p for p in _ALL_POSTS if p.get("category", DEFAULT_CATEGORY) == name]
    return jsonify(results)


@app.route("/api/sync/status")
def sync_status():
    """Returns current in-memory state — no R2 round-trip."""
    with _cache_lock:
        total = len(_ALL_POSTS)
    return jsonify({"total": total})


@app.route("/all")
def all_comics():
    with _cache_lock:
        total = len(_ALL_POSTS)
    return render_template("all.html", r2_base_url=R2_PUBLIC_URL, total=total)


@app.route("/api/sync", methods=["POST"])
def trigger_sync():
    if SYNC_TOKEN:
        provided = request.headers.get("X-Sync-Token", "")
        if provided != SYNC_TOKEN:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
    try:
        # NOTE: Gunicorn's default worker timeout is 30s. If the sync takes
        # longer, Gunicorn will kill this worker before subprocess.run returns.
        # Set --timeout on Gunicorn (e.g. gunicorn --timeout 180 app:app) if
        # syncs are expected to run longer than 30 seconds.
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "image_downloader.py"),
                "--incremental",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            error_detail = (
                result.stderr.strip()[-500:] or "Downloader exited with non-zero status"
            )
            app.logger.error("Sync subprocess failed: %s", error_detail)
            return jsonify({"ok": False, "error": error_detail}), 500

        try:
            _refresh_cache()
        except Exception as cache_err:
            # Sync ran successfully but cache refresh failed — report both.
            app.logger.error("Cache refresh after sync failed: %s", cache_err)
            return jsonify(
                {
                    "ok": False,
                    "error": f"Sync completed but cache refresh failed: {cache_err}",
                }
            ), 500

        with _cache_lock:
            total = len(_ALL_POSTS)
        return jsonify(
            {
                "ok": True,
                "total": total,
                "output": result.stdout[-500:] if result.stdout else "",
            }
        )
    except Exception as e:
        app.logger.error("Sync error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")

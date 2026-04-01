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

load_dotenv()

app = Flask(__name__)

R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "work-chronicles-storage")
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
SYNC_TOKEN = os.environ.get("SYNC_TOKEN", "")
BATCH_SIZE = 40
POSTS_KEY = "posts.json"  # shared constant — also used by image_downloader.py

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
    """
    try:
        res = s3.get_object(Bucket=R2_BUCKET, Key=POSTS_KEY)
        data = json.loads(res["Body"].read())
        posts = list(data.get("posts", {}).values())
        posts.sort(key=lambda p: p.get("post_date", ""), reverse=True)
        return posts
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
        posts = _ALL_POSTS
    first_batch = posts[:BATCH_SIZE]
    return render_template(
        "index.html",
        posts=first_batch,
        r2_base_url=R2_PUBLIC_URL,
    )


@app.route("/api/images")
def get_images_api():
    offset = int(request.args.get("offset", 0))
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


@app.route("/api/sync/status")
def sync_status():
    """Returns current in-memory state — no R2 round-trip."""
    with _cache_lock:
        total = len(_ALL_POSTS)
    return jsonify({"total": total})


@app.route("/api/sync", methods=["POST"])
def trigger_sync():
    if SYNC_TOKEN:
        provided = request.headers.get("X-Sync-Token", "")
        if provided != SYNC_TOKEN:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
    try:
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
            return jsonify(
                {
                    "ok": False,
                    "error": result.stderr[-500:]
                    or "Downloader exited with non-zero status",
                }
            ), 500
        _refresh_cache()
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
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

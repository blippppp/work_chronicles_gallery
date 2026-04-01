"""Tests for work_chronicles_gallery.

Run with:
    pytest tests.py -v
"""

import json
import os
import unittest
from unittest.mock import MagicMock, call, patch

# Provide dummy env vars before importing app so it doesn't crash on startup.
os.environ.setdefault("R2_ACCOUNT_ID", "test-account")
os.environ.setdefault("R2_BUCKET", "test-bucket")
os.environ.setdefault("R2_ACCESS_KEY", "test-key")
os.environ.setdefault("R2_SECRET_KEY", "test-secret")
os.environ.setdefault("R2_PUBLIC_URL", "https://cdn.example.com")
os.environ.setdefault("SYNC_TOKEN", "")

from botocore.exceptions import ClientError as _ClientError


def _no_such_key(*args, **kwargs):
    """Simulate R2 returning NoSuchKey — used during module-level _refresh_cache()."""
    raise _ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
    )


# Patch s3.get_object before app is imported so _refresh_cache() at module level
# doesn't make a real network call with dummy credentials.
with patch("boto3.client") as mock_boto:
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = _no_such_key
    mock_boto.return_value = mock_s3
    import app as flask_app  # noqa: E402
    import image_downloader  # noqa: E402

# After import, restore the real s3 client reference on the module so individual
# tests can patch it cleanly via patch.object(flask_app.s3, ...).
flask_app.s3 = mock_s3
image_downloader.s3 = mock_s3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_POSTS = {
    "last_sync": "2026-01-01T00:00:00+00:00",
    "posts": {
        "2": {
            "id": 2,
            "title": "Second Post",
            "slug": "second",
            "post_date": "2026-01-02",
            "canonical_url": "",
            "ext": "png",
        },
        "1": {
            "id": 1,
            "title": "First Post",
            "slug": "first",
            "post_date": "2026-01-01",
            "canonical_url": "",
            "ext": "png",
        },
    },
}


def _make_r2_response(data: dict):
    """Return a mock boto3 get_object response with the given JSON payload."""
    body = MagicMock()
    body.read.return_value = json.dumps(data).encode()
    return {"Body": body}


def _sorted_sample_posts():
    """Return SAMPLE_POSTS values sorted newest-first, matching load_posts output."""
    posts = list(SAMPLE_POSTS["posts"].values())
    posts.sort(key=lambda p: p.get("post_date", ""), reverse=True)
    return posts


# ---------------------------------------------------------------------------
# app.py — load_posts()
# ---------------------------------------------------------------------------


class TestLoadPosts(unittest.TestCase):
    def test_returns_sorted_posts_newest_first(self):
        with patch.object(
            flask_app.s3, "get_object", return_value=_make_r2_response(SAMPLE_POSTS)
        ):
            posts = flask_app.load_posts()

        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]["id"], 2, "Newest post should come first")
        self.assertEqual(posts[1]["id"], 1)

    def test_no_such_key_returns_empty(self):
        err = _ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        with patch.object(flask_app.s3, "get_object", side_effect=err):
            posts = flask_app.load_posts()

        self.assertEqual(posts, [])

    def test_unexpected_client_error_raises(self):
        err = _ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}, "GetObject"
        )
        with patch.object(flask_app.s3, "get_object", side_effect=err):
            with self.assertRaises(_ClientError):
                flask_app.load_posts()

    def test_corrupted_json_raises(self):
        body = MagicMock()
        body.read.return_value = b"not valid json {"
        with patch.object(flask_app.s3, "get_object", return_value={"Body": body}):
            with self.assertRaises(json.JSONDecodeError):
                flask_app.load_posts()


# ---------------------------------------------------------------------------
# app.py — Flask API routes
# ---------------------------------------------------------------------------


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        # Pre-populate in-memory cache with known data using load_posts logic.
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = _sorted_sample_posts()

    def tearDown(self):
        # Reset cache after each test to avoid state leaking between tests.
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = []

    def test_index_route_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Work Chronicles Gallery", res.data)

    def test_index_route_contains_r2_base_url(self):
        res = self.client.get("/")
        self.assertIn(b"cdn.example.com", res.data)

    def test_images_api_returns_batch(self):
        res = self.client.get("/api/images?offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)

    def test_images_api_offset(self):
        res = self.client.get("/api/images?offset=1")
        data = res.get_json()
        # offset=1 on a 2-item list should return 1 item
        self.assertEqual(len(data), 1)

    def test_images_api_offset_past_end(self):
        res = self.client.get("/api/images?offset=100")
        data = res.get_json()
        self.assertEqual(data, [])

    def test_images_api_invalid_offset_returns_400(self):
        res = self.client.get("/api/images?offset=abc")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_images_api_negative_offset_clamped_to_zero(self):
        res = self.client.get("/api/images?offset=-5")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        # Negative offset should be clamped to 0 — returns from the start
        self.assertEqual(len(data), 2)

    def test_search_returns_matching_posts(self):
        res = self.client.get("/api/search?q=first")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["title"], "First Post")

    def test_search_is_case_insensitive(self):
        res = self.client.get("/api/search?q=FIRST")
        data = res.get_json()
        self.assertEqual(len(data), 1)

    def test_search_empty_query_returns_empty(self):
        res = self.client.get("/api/search?q=")
        data = res.get_json()
        self.assertEqual(data, [])

    def test_search_no_match_returns_empty(self):
        res = self.client.get("/api/search?q=zzznomatch")
        data = res.get_json()
        self.assertEqual(data, [])

    def test_sync_status_uses_in_memory_state(self):
        # Should NOT hit R2 — if s3.get_object is called, the test fails.
        with patch.object(
            flask_app.s3,
            "get_object",
            side_effect=AssertionError("R2 should not be called"),
        ):
            res = self.client.get("/api/sync/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["total"], 2)

    def test_sync_requires_token_when_configured(self):
        original = flask_app.SYNC_TOKEN
        try:
            flask_app.SYNC_TOKEN = "secret"
            res = self.client.post("/api/sync")
            self.assertEqual(res.status_code, 401)
        finally:
            flask_app.SYNC_TOKEN = original

    def test_sync_accepts_valid_token(self):
        original = flask_app.SYNC_TOKEN
        try:
            flask_app.SYNC_TOKEN = "secret"
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "ok"
            with (
                patch("subprocess.run", return_value=mock_result),
                patch.object(flask_app, "_refresh_cache") as mock_refresh,
            ):
                res = self.client.post("/api/sync", headers={"X-Sync-Token": "secret"})
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.get_json()["ok"])
            # Verify _refresh_cache was actually called after successful sync.
            mock_refresh.assert_called_once()
        finally:
            flask_app.SYNC_TOKEN = original

    def test_sync_returns_500_on_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "something went wrong"
        with patch("subprocess.run", return_value=mock_result):
            res = self.client.post("/api/sync")
        self.assertEqual(res.status_code, 500)
        self.assertFalse(res.get_json()["ok"])

    def test_sync_returns_500_if_cache_refresh_fails(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        with (
            patch("subprocess.run", return_value=mock_result),
            patch.object(flask_app, "_refresh_cache", side_effect=Exception("R2 down")),
        ):
            res = self.client.post("/api/sync")
        self.assertEqual(res.status_code, 500)
        data = res.get_json()
        self.assertFalse(data["ok"])
        self.assertIn("cache refresh failed", data["error"])

    def test_sync_unauthenticated_when_token_empty(self):
        original = flask_app.SYNC_TOKEN
        try:
            flask_app.SYNC_TOKEN = ""
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "ok"
            with (
                patch("subprocess.run", return_value=mock_result),
                patch.object(flask_app, "_refresh_cache"),
            ):
                # No token header — should succeed when SYNC_TOKEN is empty
                res = self.client.post("/api/sync")
            self.assertEqual(res.status_code, 200)
        finally:
            flask_app.SYNC_TOKEN = original

    def test_categories_endpoint_returns_list(self):
        res = self.client.get("/api/categories")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, list)
        # Each entry must have required keys
        for cat in data:
            self.assertIn("name", cat)
            self.assertIn("icon", cat)
            self.assertIn("count", cat)

    def test_categories_counts_match_cache(self):
        # Inject posts with known categories
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = [
                {"id": 1, "title": "A", "category": "Meetings", "ext": "jpg"},
                {"id": 2, "title": "B", "category": "Meetings", "ext": "jpg"},
                {"id": 3, "title": "C", "category": "Productivity", "ext": "jpg"},
            ]
        res = self.client.get("/api/categories")
        data = res.get_json()
        counts = {c["name"]: c["count"] for c in data}
        self.assertEqual(counts.get("Meetings"), 2)
        self.assertEqual(counts.get("Productivity"), 1)

    def test_category_filter_endpoint(self):
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = [
                {"id": 1, "title": "A", "category": "Meetings", "ext": "jpg"},
                {"id": 2, "title": "B", "category": "Productivity", "ext": "jpg"},
            ]
        res = self.client.get("/api/category/Meetings")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["category"], "Meetings")

    def test_category_filter_unknown_returns_empty(self):
        res = self.client.get("/api/category/DoesNotExist")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), [])


# ---------------------------------------------------------------------------
# image_downloader.py — safe_get() retry logic
# ---------------------------------------------------------------------------


class TestSafeGet(unittest.TestCase):
    def test_returns_response_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(image_downloader._session, "get", return_value=mock_resp):
            result = image_downloader.safe_get("http://example.com")
        self.assertEqual(result, mock_resp)

    def test_returns_none_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch.object(image_downloader._session, "get", return_value=mock_resp):
            result = image_downloader.safe_get("http://example.com")
        self.assertIsNone(result)

    def test_retries_on_429(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        with (
            patch.object(
                image_downloader._session, "get", side_effect=[resp_429, resp_200]
            ),
            patch("time.sleep"),
        ):
            result = image_downloader.safe_get("http://example.com", retries=3)
        self.assertEqual(result, resp_200)

    def test_returns_none_after_all_retries_exhausted_on_exception(self):
        with (
            patch.object(
                image_downloader._session, "get", side_effect=ConnectionError("timeout")
            ),
            patch("time.sleep"),
        ):
            result = image_downloader.safe_get("http://example.com", retries=2)
        self.assertIsNone(result)

    def test_returns_none_after_all_retries_exhausted_on_429(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        with (
            patch.object(image_downloader._session, "get", return_value=resp_429),
            patch("time.sleep"),
        ):
            result = image_downloader.safe_get("http://example.com", retries=3)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# image_downloader.py — load_metadata()
# ---------------------------------------------------------------------------


class TestLoadMetadata(unittest.TestCase):
    def test_returns_empty_on_no_such_key(self):
        err = _ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        with patch.object(image_downloader.s3, "get_object", side_effect=err):
            result = image_downloader.load_metadata()
        self.assertEqual(result["posts"], {})
        self.assertIsNone(result["last_sync"])

    def test_raises_on_access_denied(self):
        err = _ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}, "GetObject"
        )
        with patch.object(image_downloader.s3, "get_object", side_effect=err):
            with self.assertRaises(_ClientError):
                image_downloader.load_metadata()

    def test_raises_on_corrupted_json(self):
        body = MagicMock()
        body.read.return_value = b"{{not json"
        with patch.object(
            image_downloader.s3, "get_object", return_value={"Body": body}
        ):
            with self.assertRaises(json.JSONDecodeError):
                image_downloader.load_metadata()


# ---------------------------------------------------------------------------
# app.py — /all route
# ---------------------------------------------------------------------------


class TestAllRoute(unittest.TestCase):
    def setUp(self):
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = _sorted_sample_posts()

    def tearDown(self):
        with flask_app._cache_lock:
            flask_app._ALL_POSTS = []

    def test_all_route_returns_200(self):
        res = self.client.get("/all")
        self.assertEqual(res.status_code, 200)

    def test_all_route_contains_r2_base_url(self):
        res = self.client.get("/all")
        self.assertIn(b"cdn.example.com", res.data)

    def test_all_route_shows_total_count(self):
        res = self.client.get("/all")
        # Template renders "All 2 comics" with our 2-post sample cache
        self.assertIn(b"2", res.data)

    def test_index_has_link_to_all(self):
        res = self.client.get("/")
        self.assertIn(b"/all", res.data)


if __name__ == "__main__":
    unittest.main()

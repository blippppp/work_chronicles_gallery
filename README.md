# Work Chronicles Gallery

A self-hosted image gallery for [Work Chronicles](https://www.workchronicles.com) comics.  
Images are fetched from the Work Chronicles API and stored in **Cloudflare R2**. The Flask app serves them from R2's public CDN — no local image storage needed.

---

## Architecture

```
workchronicles.com API
        │
        ▼
image_downloader.py  ──── uploads ──▶  Cloudflare R2
                                           │
                              posts.json ──┘
                                           │
                              Flask app reads posts.json from R2
                              and serves the gallery UI
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A Cloudflare R2 bucket with a **public URL** enabled
- R2 API credentials (Account ID, Access Key, Secret Key)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```
R2_ACCOUNT_ID=your_account_id
R2_BUCKET=your-bucket-name
R2_ACCESS_KEY=your_access_key_id
R2_SECRET_KEY=your_secret_access_key
R2_PUBLIC_URL=https://pub-xxxx.r2.dev
SYNC_TOKEN=a_random_secret_for_the_sync_endpoint
```

> `SYNC_TOKEN` protects the `POST /api/sync` endpoint. Set it to any random string.  
> If omitted, the endpoint is unauthenticated (fine for local dev only).

---

## Syncing images

### First run (full sync)

Downloads all posts and uploads images to R2:

```bash
python image_downloader.py --full
```

### Subsequent runs (incremental sync)

Fetches only posts published since the last sync:

```bash
python image_downloader.py --incremental
```

### Auto-detect (default)

If `posts.json` does not exist in R2, runs a full sync; otherwise incremental:

```bash
python image_downloader.py
```

---

## Running the web app

### Development

```bash
python app.py
```

Starts Flask on `http://localhost:5000` with the Werkzeug dev server.

### Production

Use Gunicorn (included in `requirements.txt`):

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:8000
```

Or via the `Procfile`:

```bash
heroku local   # or any Procfile-compatible runner
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Gallery UI |
| `GET` | `/api/images?offset=N` | Paginated post list (40 per page) |
| `GET` | `/api/search?q=query` | Title search |
| `GET` | `/api/sync/status` | Current post count and last sync time |
| `POST` | `/api/sync` | Trigger incremental sync (requires `X-Sync-Token` header if `SYNC_TOKEN` is set) |

**Example sync call with token:**

```bash
curl -X POST http://localhost:5000/api/sync \
     -H "X-Sync-Token: your_sync_token"
```

---

## Features

- **Infinite scroll** gallery with lazy-loaded images
- **Search** by title (case-insensitive)
- **Lightbox viewer** with keyboard navigation (←/→/Esc)
- **Parallel image uploading** with rate-limit handling
- **Incremental sync** — only new posts are downloaded

---

## Project structure

```
work_chronicles_gallery/
├── app.py                 # Flask application
├── image_downloader.py    # API fetcher + R2 uploader
├── requirements.txt       # Python dependencies
├── Procfile               # Gunicorn production entrypoint
├── templates/
│   └── index.html         # Frontend (vanilla HTML/CSS/JS)
└── .env                   # Local secrets (never commit)
```

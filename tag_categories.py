"""One-time script to add categories to existing posts.json in R2."""

import os, json, logging
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from categories import categorize_posts, get_category_summary

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "work-chronicles-storage")
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4", region_name="auto"),
    region_name="auto",
)

log.info("Loading posts.json from R2...")
res = s3.get_object(Bucket=R2_BUCKET, Key="posts.json")
data = json.loads(res["Body"].read())

posts_list = list(data["posts"].values())
log.info("Loaded %d posts", len(posts_list))

# Check how many already have categories
already_tagged = sum(1 for p in posts_list if p.get("category"))
log.info("Already tagged: %d", already_tagged)

# Categorize all posts (skips already-tagged ones)
categorize_posts(posts_list)

# Rebuild the posts dict
data["posts"] = {str(p["id"]): p for p in posts_list}

# Save back to R2
log.info("Saving updated posts.json to R2...")
s3.put_object(
    Bucket=R2_BUCKET,
    Key="posts.json",
    Body=json.dumps(data, indent=2),
    ContentType="application/json",
)

# Print summary
summary = get_category_summary(posts_list)
log.info("\nCategory distribution:")
for cat, count in summary.items():
    log.info("  %-25s %4d", cat, count)
log.info("\nDone. %d posts categorized.", len(posts_list))

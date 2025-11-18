# main.py
from fastapi import FastAPI, Query
from pymongo import MongoClient
from transformers import pipeline
import threading, time, math
from bson import ObjectId
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

app = FastAPI()

# ---------------- CONFIG ----------------
MONGO_URI = "mongodb+srv://db_user:WcoEqUnPZTKwACY8@cluster0.5iz2xl.mongodb.net/?appName=Cluster0"
DB = "social_media"
COL = "posts"

# Fast keyword map
KEYWORD_MAP = {
    "Cricket": ["cricket", "ipl", "t20", "batsman", "bowler", "wicket"],
    "Football": ["football", "soccer", "goal", "fifa"],
    "Technology": ["tech", "technology", "computer", "ai", "python", "javascript"],
    "Business": ["business", "startup", "market", "stock", "store", "logo"],
    "Love": ["love", "make love", "miss you", "romance"],
    "Friendship": ["friend", "friends", "buddy"],
    "Entertainment": ["movie", "film", "song", "music", "concert"],
    "Food": ["food", "pizza", "burger", "panjabi", "restaurant"],
    "Travel": ["travel", "trip", "vacation", "flight"],
    "Photography": ["sunset", "photo", "cozy", "workspace", "coffee", "rainy"]
}

CANDIDATE_LABELS = list(KEYWORD_MAP.keys()) + ["News", "Fashion", "Health", "Education"]

MODEL_NAME = "valhalla/distilbart-mnli-12-1"

# weights
W_POST = 1
W_LIKE = 1
W_COMMENT = 2
W_REPLY = 2
W_SHARE = 1

# batch size
BATCH_SIZE = 64

# ---------------- INIT ----------------
client = MongoClient(MONGO_URI)
collection = client[DB][COL]
print("✅ Connected to MongoDB")

print("🔄 Loading zero-shot model...")
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
print("✅ Model loaded")

user_interests: Dict[str, Dict[str, int]] = {}
topic_totals: Dict[str, int] = {}
last_recompute_ts: Optional[datetime] = None


# ---------------- helpers ----------------

def convert_objectid_recursive(obj: Any) -> Any:
    if isinstance(obj, list):
        return [convert_objectid_recursive(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_objectid_recursive(v) for k, v in obj.items()}
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

def text_for_post(post: dict) -> str:
    parts = []
    content = post.get("content") or post.get("text") or ""
    parts.append(str(content))

    img = post.get("imageData") or post.get("image") or {}
    if isinstance(img, dict):
        for k in ("caption", "alt", "title", "description"):
            if img.get(k):
                parts.append(str(img.get(k)))

    return " ".join(p for p in parts if p and str(p).strip())

def keyword_detect(text: str) -> Optional[str]:
    """Return None if no keyword match (NO 'Other')"""
    if not text:
        return None
    t = text.lower()
    for label, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in t:
                return label
    return None

def add_score(store: Dict[str, Dict[str, int]], uid: Optional[str], topic: str, score: int):
    if not uid or not topic:
        return
    uid = str(uid)
    store.setdefault(uid, {})
    store[uid][topic] = store[uid].get(topic, 0) + score
    topic_totals[topic] = topic_totals.get(topic, 0) + score

def process_comment_recursive(store: Dict[str, Dict[str, int]], comment: dict, topic: str):
    try:
        commenter = comment.get("userId")
        add_score(store, commenter, topic, W_COMMENT)

        for like in comment.get("likes", []):
            add_score(store, like.get("userId"), topic, W_LIKE)

        for reply in comment.get("replies", []):
            add_score(store, reply.get("userId"), topic, W_REPLY)
            process_comment_recursive(store, reply, topic)
    except Exception:
        pass


# ---------------- FAST PASS (keywords only) ----------------
def recompute_interests_fast(posts: List[dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str,int]]:
    new_interests: Dict[str, Dict[str, int]] = {}
    new_topic_totals: Dict[str,int] = {}

    for post in posts:
        try:
            pu = str(post.get("userId"))
            txt = text_for_post(post)
            kw = keyword_detect(txt)

            if not kw:
                continue    # skip unknown topics

            topic = kw

            add_score(new_interests, pu, topic, W_POST)
            new_topic_totals[topic] = new_topic_totals.get(topic,0) + W_POST

            for like in post.get("likes", []):
                add_score(new_interests, like.get("userId"), topic, W_LIKE)

            for comment in post.get("comments", []):
                process_comment_recursive(new_interests, comment, topic)

            shares = post.get("shares_data") or []
            for s in shares:
                add_score(new_interests, s.get("userId"), topic, W_SHARE)

        except Exception:
            continue

    return new_interests, new_topic_totals


# ---------------- ZERO-SHOT PASS ----------------
def recompute_interests_batch_zero_shot(posts: List[dict]) -> Tuple[Dict[str, Dict[str, int]], Dict[str,int]]:
    new_interests: Dict[str, Dict[str, int]] = {}
    new_topic_totals: Dict[str,int] = {}

    texts = []
    meta = []

    for post in posts:
        pu = str(post.get("userId"))
        txt = text_for_post(post)
        kw = keyword_detect(txt)

        texts.append(txt)
        meta.append((pu, post, kw))

    total = len(texts)
    if total == 0:
        return new_interests, new_topic_totals

    batches = math.ceil(total / BATCH_SIZE)

    for i in range(batches):
        start = i * BATCH_SIZE
        end = min(total, start + BATCH_SIZE)

        batch_texts = texts[start:end]
        batch_meta = meta[start:end]

        try:
            res = classifier(batch_texts, CANDIDATE_LABELS, multi_label=False)

            for j, r in enumerate(res):
                pu, post, kw = batch_meta[j]

                if kw:
                    topic = kw
                else:
                    topic = r["labels"][0]   # no "Other"

                add_score(new_interests, pu, topic, W_POST)
                new_topic_totals[topic] = new_topic_totals.get(topic,0) + W_POST

                for like in post.get("likes", []):
                    add_score(new_interests, like.get("userId"), topic, W_LIKE)

                for comment in post.get("comments", []):
                    process_comment_recursive(new_interests, comment, topic)

                shares = post.get("shares_data") or []
                for s in shares:
                    add_score(new_interests, s.get("userId"), topic, W_SHARE)

        except Exception as e:
            print("Zero-shot batch error:", e)

    return new_interests, new_topic_totals


# ---------------- FULL RECOMPUTE ----------------
def recompute_full(refine_with_zero_shot: bool = True):
    global user_interests, topic_totals, last_recompute_ts

    posts = list(collection.find({}))

    # fast-pass (no other)
    new_interests, new_topic_totals = recompute_interests_fast(posts)

    if refine_with_zero_shot:
        print("🔁 Zero-shot refine running...")
        zs_interests, zs_totals = recompute_interests_batch_zero_shot(posts)
        new_interests, new_topic_totals = zs_interests, zs_totals

    user_interests = new_interests
    topic_totals = new_topic_totals
    last_recompute_ts = datetime.utcnow()

    print(f"✅ Recompute done: users={len(user_interests)} topics={len(topic_totals)}")
    return len(user_interests), len(topic_totals)


# ---------------- BACKGROUND THREAD ----------------
def background_worker(interval: int = 30):
    while True:
        try:
            print("🔄 Background refine...")
            recompute_full(refine_with_zero_shot=True)
            time.sleep(interval)
        except Exception as e:
            print("Background error:", e)
            time.sleep(5)


# ---------------- API ROUTES ----------------
@app.get("/")
def home():
    return {"status": "ok", "message": "Interest service running"}

@app.get("/debug-posts")
def debug_posts(limit: int = Query(10)):
    posts = list(collection.find({}).limit(limit))
    return {"status":"ok", "count": len(posts), "posts": convert_objectid_recursive(posts)}

@app.get("/recompute-now")
def recompute_now(refine: bool = True):
    u, t = recompute_full(refine_with_zero_shot=refine)
    return {"status":"ok", "users": u, "topics": t}

@app.get("/interests")
def get_interests(top: int = Query(3), full_map: bool = False):
    result = {}
    for uid, scores in user_interests.items():
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result[uid] = {"top": sorted_topics[:top]}
        if full_map:
            result[uid]["map"] = scores

    return {"status":"ok", "users": len(result), "user_interests": result}

@app.get("/status")
def status():
    return {"status":"ok", "users": len(user_interests), "topics": len(topic_totals)}


# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    print("🚀 Startup keyword-only recompute...")
    recompute_full(refine_with_zero_shot=False)

    print("🚀 Starting background refinements...")
    thread = threading.Thread(target=background_worker, args=(30,), daemon=True)
    thread.start()

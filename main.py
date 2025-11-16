from fastapi import FastAPI, BackgroundTasks
from pymongo import MongoClient
from transformers import pipeline
import threading
import time

app = FastAPI()

# -------------------------------------
# 1. Load lightweight Zero-Shot Model
# -------------------------------------
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

candidate_labels = [
    "Cricket", "Football", "government", "politics",
    "Love", "Friendship", "Technology", "Business",
    "Entertainment", "News"
]

# -------------------------------------
# 2. DB connection
# -------------------------------------
client = MongoClient("mongodb+srv://db_user:WcoEqUnPZTKwACY8@cluster0.5iz2xl.mongodb.net/")
db = client["social_media"]
collection = db["posts"]

# -------------------------------------
# 3. User interest storage
# -------------------------------------
user_interests = {}

# -------------------------------------
# 4. Detect topic
# -------------------------------------
def detect_topic(text: str):
    if not text:
        return "Unknown"
    result = classifier(text, candidate_labels)
    return result["labels"][0]

# -------------------------------------
# 5. Background job (safe for Render)
# -------------------------------------
def background_interest_job():
    global user_interests

    while True:
        print("Updating interests...")
        new_interests = {}

        posts = list(collection.find({}))

        for post in posts:
            user = str(post.get("userId"))
            content = post.get("content", "")
            likes = post.get("likes", [])
            comments = post.get("comments", [])

            topic = detect_topic(content)

            # Own post
            new_interests.setdefault(user, {})
            new_interests[user][topic] = new_interests[user].get(topic, 0) + 1

            # Likes
            for like in likes:
                lid = str(like.get("userId"))
                new_interests.setdefault(lid, {})
                new_interests[lid][topic] = new_interests[lid].get(topic, 0) + 1

            # Comments
            for comment in comments:
                cid = str(comment.get("userId"))
                new_interests.setdefault(cid, {})
                new_interests[cid][topic] = new_interests[cid].get(topic, 0) + 2

        user_interests = new_interests
        time.sleep(30)     # Run every 30 seconds


# -------------------------------------
# 6. Start background thread when app starts
# -------------------------------------
@app.on_event("startup")
def start_background_task():
    thread = threading.Thread(target=background_interest_job, daemon=True)
    thread.start()


# -------------------------------------
# 7. Routes
# -------------------------------------
@app.get("/")
def home():
    return {"message": "User Interest Service Running 🚀"}

@app.get("/interests")
def get_all_interests():
    return user_interests

@app.get("/interests/{user_id}")
def get_user_interest(user_id: str):
    return user_interests.get(user_id, {})

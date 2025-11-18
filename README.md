🚀 User Interest Detection API

A FastAPI + MongoDB + Zero-Shot AI service that detects user interests based on posts, likes, and comments.

🔧 Setup (Local)
Install
pip install -r requirements.txt

Run
uvicorn main:app --reload --host 0.0.0.0 --port 8000

🌐 Deploy on Render

Build Command

pip install -r requirements.txt


Start Command

uvicorn main:app --host 0.0.0.0 --port $PORT

📡 API Endpoints

Home: GET /
All Interests: GET /interests
User Interest: GET /interests/{user_id}

🧠 How It Works

Background thread runs every 30 seconds

Fetches posts from MongoDB

Detects topic using Zero-Shot model

Likes = +1, Comments = +2, Own posts = +1

Stores results in memory (user_interests)

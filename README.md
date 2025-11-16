User Interest API

FastAPI service that analyzes social media posts and tracks user interests using AI.
🚀 Quick Start
bash

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 9000

📌 API Endpoints

    GET / - Health check

    GET /interests - All users' interests

    GET /interests/{user_id} - Specific user's interests

🌐 Access

Main URL: http://localhost:9000
API Docs: http://localhost:9000/docs
🔧 Features

    AI-powered topic detection (10 categories)

    Auto-updates every 30 seconds

    MongoDB integration

    Real-time interest scoring

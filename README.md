## Credit Card Advisor — React + FastAPI

This project migrates the Streamlit UI to a modern React frontend and introduces a FastAPI backend for production-friendly APIs.

### Quick start (local)

1. Install Python deps and start API:

```
python3 -m venv .venv && source .venv/bin/activate
python run.py
```

2. In another terminal, start frontend:

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Environment
- Set `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env` for LLM smalltalk/ranking and live web search.
- Optionally set `CREDIT_CARD_DATA_PATH` to your dataset CSV (defaults to `credit_cards_dataset.csv`).

### Production build (Docker)

```
docker build -t cc-advisor .
docker run -p 8000:8000 --env-file .env cc-advisor
```

The built frontend is served from `/` and the API is under `/api/*`.


### One-command run (docker-compose)

```
docker compose up --build
```

It will start the API on port 8000 and the React dev server on port 5173 with hot reload.


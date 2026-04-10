# Deployment Guide (Docker)

## 1) Prerequisites
- Docker Engine and Docker Compose plugin installed
- `GROQ_API_KEY` available

## 2) Set environment variables
You can export these in shell or put them in a `.env` file used by Docker Compose.

Required:
- `GROQ_API_KEY`

Optional:
- `GROQ_MODEL` (default: `openai/gpt-oss-20b`)
- `FAKE_NEWS_DATASET` (default: `data/fake_real_combined.csv`)

## 3) Build and run
```bash
docker compose up --build -d
```

## 4) Verify
```bash
curl http://127.0.0.1:8000/health
```

Open UI at:
- `http://127.0.0.1:8000/`

API docs at:
- `http://127.0.0.1:8000/docs`

## 5) Logs and stop
```bash
docker compose logs -f
docker compose down
```

## 6) Deploy on another server
1. Copy project to server (git clone or scp).
2. Ensure the dataset file exists at `data/fake_real_combined.csv` or set `FAKE_NEWS_DATASET` to a valid path.
3. Set `GROQ_API_KEY`.
4. Run `docker compose up --build -d`.

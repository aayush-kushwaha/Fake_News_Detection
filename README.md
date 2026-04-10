# Fake News Detection: Deep Learning + LLM-Assisted Fact Checking

A complete fake-news detection project that combines:
- Classical NLP baseline (TF-IDF + Logistic Regression)
- Deep learning experiments in a notebook (BiLSTM + DistilBERT)
- A FastAPI web app with a modern UI
- Optional Groq-powered LLM analysis and external-reference fact checking

## Features
- End-to-end notebook pipeline for model experimentation and analysis
- FastAPI backend with prediction and combined analysis endpoints
- Browser UI for article classification and explanation
- Optional LLM layer:
  - article summary
  - risk explanation
  - external evidence retrieval with references
- Dockerized deployment for local and server portability

## Project Structure

```text
Fake_News_Detection/
├─ app/
│  ├─ main.py                # FastAPI app + routes + UI serving
│  ├─ model_service.py       # Lightweight baseline classifier service
│  ├─ schemas.py             # API request/response models
│  ├─ templates/
│  │  └─ index.html          # Frontend template
│  └─ static/
│     ├─ css/styles.css      # UI styling
│     └─ js/app.js           # Frontend logic
├─ data/
│  └─ fake_real_combined.csv # Dataset used for training/inference
├─ groq_utils.py             # Groq + retrieval helper utilities
├─ final_project_fake_news.ipynb
├─ final_project_fake_news.html
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
└─ DEPLOYMENT.md
```

## Notebook Workflow

Notebook file:
- [final_project_fake_news.ipynb](final_project_fake_news.ipynb)

What it contains:
1. Data loading and cleaning (`clean_text_v2` with publisher-tag leakage mitigation)
2. Baseline model: TF-IDF + Logistic Regression
3. BiLSTM model training and evaluation
4. DistilBERT fine-tuning and comparison
5. Optional LLM layer section:
   - summary
   - risk explanation
   - external reference-backed fact check

Open in Jupyter/VS Code and run cells in order.

## Web App and API

Backend entrypoint:
- [app/main.py](app/main.py)

### Routes
- `GET /` -> UI
- `GET /health` -> service health and model status
- `POST /api/predict` -> model-only prediction
- `POST /api/analyze` -> prediction + optional LLM analysis
- `GET /docs` -> Swagger docs

### Example Request

```json
POST /api/analyze
{
  "text": "<article text>",
  "use_llm": true
}
```

### Example Response (shape)

```json
{
  "prediction": {
    "label": 1,
    "label_name": "fake",
    "confidence": 0.91
  },
  "summary": "...",
  "risk_explanation": "...",
  "fact_check": {
    "verdict": "mixed",
    "confidence": 60,
    "explanation": "...",
    "references": [
      {
        "title": "...",
        "url": "https://...",
        "snippet": "...",
        "source_type": "trusted"
      }
    ],
    "has_trusted_references": true
  },
  "warnings": []
}
```

## Local Setup (No Docker)

### 1) Create and activate environment

Example with conda:

```bash
conda create -n fake-news-detection-dl python=3.10 -y
conda activate fake-news-detection-dl
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run app

```bash
uvicorn app.main:app --reload
```

Open:
- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Environment Variables

Create a local `.env` (already ignored by git):

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
FAKE_NEWS_DATASET=data/fake_real_combined.csv
```

Notes:
- `GROQ_API_KEY` is required only for LLM features.
- Core prediction endpoint works without LLM if model is loaded.

## Docker Deployment

Files:
- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [DEPLOYMENT.md](DEPLOYMENT.md)

Quick start:

```bash
docker compose up --build -d
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

## Deployment to Another System

1. Clone the repository on target machine.
2. Ensure dataset exists at `data/fake_real_combined.csv` (or set `FAKE_NEWS_DATASET`).
3. Set `GROQ_API_KEY`.
4. Run with Docker Compose.

## Troubleshooting

### 1) `500` on `/` due to template response
Use latest code where home route calls:
- `TemplateResponse(request=request, name="index.html", context={})`

### 2) LLM retrieval issues / no references
- Ensure `ddgs` is installed.
- Restart notebook kernel if using notebook LLM section.
- External references can be empty for weak or highly noisy queries.

### 3) Import not resolved warnings in editor
Usually environment mismatch. Install in active interpreter:

```bash
pip install -r requirements.txt
```

### 4) Dataset not found
Set env var:

```bash
export FAKE_NEWS_DATASET=data/fake_real_combined.csv
```

(or Windows PowerShell equivalent)

## Tech Stack
- Python
- pandas, scikit-learn, PyTorch, transformers
- FastAPI + Uvicorn + Jinja2
- OpenAI-compatible Groq API
- DDGS retrieval for external search
- Docker + Docker Compose

## License
This repository is submitted as part of an academic deep learning course project.

Usage policy:
- Educational and evaluation use only
- No commercial use without explicit permission from the project author(s)
- Attribution required when referencing project code, reports, or results

Unless your instructor provides a different template, treat this work as
"All Rights Reserved" outside course-related usage.

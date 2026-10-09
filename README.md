# Hanami Cloud support assistant (RAG + evals)

A small end-to-end AI product: a support chatbot that answers questions from documentation, shows its sources, collects thumbs up/down feedback, logs every query, and is measured with an LLM-as-Judge eval set.

The docs in `backend/docs` are fictional (a made-up file storage service, with an English set and a Japanese FAQ). Replace them with any public documentation to reuse the project.

<!-- Live demo: add your link here after deploying, e.g. **Live demo:** https://your-app.vercel.app -->

## What it includes

1. **Backend (Python, FastAPI):** loads documents, splits them into chunks, embeds them and stores them in Chroma, then answers questions with Claude or Gemini using the retrieved chunks and numbered citations.
2. **Frontend (React + TypeScript):** a chat page that shows the answer, the source passages, helpful / not helpful buttons, and a small usage panel.
3. **Evals:** `backend/run_eval.py` runs 21 question/answer pairs (including 4 in Japanese and 1 unanswerable question) through several configurations and scores each answer for faithfulness, relevance and correctness with an LLM judge (the same model you configured).
4. **Deployment:** Dockerfile for the backend, a GitHub Actions workflow that checks both halves on every push.

## Architecture

```
React + TypeScript  -->  FastAPI  -->  Chroma (retrieve top chunks)
   (chat + stats)          |     -->  Claude or Gemini (answer with citations)
                           |
                           +--> SQLite (queries, latency, tokens, feedback)
```

## Run it locally

You need Python 3.11+, Node 20+ and one API key: a Google Gemini key (free from [Google AI Studio](https://aistudio.google.com/api-keys)) or an Anthropic key. Whichever key you set is used automatically; set `LLM_PROVIDER=google` or `anthropic` to choose explicitly.

```bash
# Backend
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here         # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
# (using Claude instead? set ANTHROPIC_API_KEY the same way)
uvicorn main:app --reload --port 8000

# Frontend (in a second terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and ask something like "How long are deleted files kept on the Plus plan?"

## Evaluation

```bash
cd backend
python run_eval.py
```

This compares three configurations (chunk size 300 vs 800, and prompt v1 vs v2) and writes `eval_results.md`.

Results from an earlier run (these two configs came from a previous version of the eval, so the names differ from the current `CONFIGS` in `run_eval.py`). Re-run `python run_eval.py` and paste the new `eval_results.md` table here:

| Config | Faithfulness | Relevance | Correctness | Avg latency (ms) |
|---|---|---|---|---|
| chunk300_k3_baseline | 100% | 96% | 92% | 1769 |
| chunk300_k3_headings | 100% | 92% | 92% | 1703 |

## What broke and how I fixed it

**Retrieval misses when the product name is in the question.** In a quick test, the search found the correct chunk for "What encryption is used?" and "How are files encrypted?", but not for "What encryption does Hanami Cloud use?". The product name appears in many chunks, so it pulls in generic passages ahead of the one about encryption. The model behaved correctly: it refused to guess from passages that did not contain the answer. Ideas to fix it: strip the product name from the query before searching, or add a keyword or hybrid search step.

**Still to check (Japanese retrieval).** The default embedding model is English-only, so Japanese retrieval may be weaker. To test a fix, set `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`, run `pip install sentence-transformers`, re-run the eval and compare the tables. Add what you find here.

## Deploy

Backend on Google Cloud Run:

```bash
gcloud run deploy hanami-api --source backend --region asia-northeast1 \
  --allow-unauthenticated --set-env-vars GEMINI_API_KEY=your-key,CORS_ORIGINS=https://your-frontend-url
```

Frontend on Vercel or Netlify: set the project folder to `frontend`, add the environment variable `VITE_API_URL` with your backend URL, and deploy.

Note: SQLite on Cloud Run is wiped when the container restarts, so stats are not permanent in this version. Moving the database to a managed Postgres is the next step.

## Limits and next steps

1. Answers are not streamed yet.
2. The eval set is small (21 questions); grow it to 40+ before trusting small differences between configurations.
3. The judge model can be biased, so spot-check a few judged answers by hand.
4. Add agent tools (for example a mock ticket lookup and a "hand off to a human" action).

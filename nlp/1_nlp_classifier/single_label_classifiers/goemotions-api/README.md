# GoEmotions Classifier API

A minimal FastAPI wrapper around the fine-tuned `bert_goemotions_v1` model.

## 1. Get your model files in place

Copy your saved model folder (tokenizer + model weights + `label_encoder.pkl`)
into this directory, named `bert_goemotions_v1/`, so the layout looks like:

```
goemotions-api/
├── main.py
├── requirements.txt
├── Dockerfile
└── bert_goemotions_v1/
    ├── config.json
    ├── model.safetensors (or pytorch_model.bin)
    ├── tokenizer.json / vocab.txt / etc.
    └── label_encoder.pkl
```

## 2. Run locally

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

# Run in terminal: http://localhost:8000/docs#/

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I cannot believe how well that went, I am so proud of myself"}'
```

Expected response shape:

```json
{"label": "pride", "confidence": 0.87, "latency_ms": 42.1}
```

Every call also appends a row to `prediction_log.jsonl` — that log is what
the next step (drift monitoring) will read from.

## 3. Run with Docker

```bash
docker build -t goemotions-api .
docker run -p 8000:8000 goemotions-api
```

## 4. Definitions to Note

- **Latency**: run a handful of requests and note the `latency_ms` field.
  CPU-only inference for a single BERT-base call is typically in the tens
  of milliseconds; if you want to speak to optimization, mention batching
  requests or exporting to ONNX as next steps.
- **Statelessness**: the model loads once at startup, not per-request —
  worth calling out, since reloading weights per request is a common
  mistake that tanks latency.
- **Logging**: this is intentionally simple (a JSONL file) so the next
  step — drift detection — has something concrete to read from.

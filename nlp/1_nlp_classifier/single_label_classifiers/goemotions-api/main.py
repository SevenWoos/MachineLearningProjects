"""
FastAPI wrapper for the fine-tuned bert_goemotions_v1 SINGLE emotion classifier.

Run locally:
    uvicorn main:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "I cannot believe how well that went, I am so proud of myself"}'
        
Run in browser:
    http://localhost:8000/docs#/
"""

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import BertForSequenceClassification, BertTokenizerFast

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_DIR = Path("bert_goemotions_v1")   # adjust if your saved model lives elsewhere
LOG_PATH = Path("prediction_log.jsonl")   # every prediction gets appended here
MAX_LENGTH = 128                           # match whatever I trained with
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Load model artifacts once, at startup
# --------------------------------------------------------------------------

# Loading BERT weights from disk can take time. So rather than load it everytime we run the "/predict" function(every single request gets expensive), we run it ONCE at import time - when uvicorn starts the app NOT per-request.
app = FastAPI(title="GoEmotions Classifier API")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)

# switches off dropout and batch-norm-style training behavior so inference is deterministic. 
# # if skipped, predictions become slightly random/inconsistent because dropout is still randomly zeroing out neurons.
model.eval()

# Loaded ONCE too, for the same reason — it's just a lookup table mapping model output indices (0-27) back to human-readable emotion names.
with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

# FastAPI uses this to auto-validate incoming JSON.
class PredictRequest(BaseModel):
    # Guardrails against empty strings and absurdly long payloads.
    text: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


# --------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------

# The acutal prediction logic.
def run_inference(text: str) -> tuple[str, float, float]:
    # Start a timer to get the latency_ms.
    start = time.perf_counter()

    # Tokenize to convert raw text to token IDs for BERT. truncate and pad and move tensors to CPU/GPU.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    # Tells PyTorch not to track gradients, since we're NOT training. Saves memory and speeds up inference.
    with torch.no_grad():
        # Forward pass that runs the actual Transformer and returns the raw logits.
        logits = model(**inputs).logits
        # Softmax converts logits into probabilities.
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)

    # "inverse_transform" maps numeric calss index back to string like "anger".
    label = label_encoder.inverse_transform([pred_idx.item()])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    return label, confidence.item(), latency_ms


def log_prediction(text: str, label: str, confidence: float, latency_ms: float) -> None:
    """Append every prediction to a local JSONL log.

    This is the raw material we'll use for drift-monitoring: once you're
    logging (timestamp, text, predicted label, confidence), you can later
    compute a rolling label distribution and confidence distribution, and
    compare each new window against a baseline window with something like
    population stability index or KL divergence.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_label": label,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        
    # Every real prediction gets APPENDED as one JSON line. This is the raw data you'll need for DRIFT DETECTION later.
    # Without a record of what the model has been seeing in production, you can't compare "today's inputs" against "training-time inputs.# # Built logging in from the start because we can't monitor what we don't record.
    
    # Terminal command if we want to peak the jsonl: cat prediction_log.jsonl


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

# The 2 endpoints.

# A standard convention in production services — load balancers, orchestrators (Kubernetes, ECS), and uptime monitors ping this to check if your service is alive, without exercising the actual model. 
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

# Validate input (automatic, via the Pydantic model), run inference, log it, return a typed response.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label, confidence, latency_ms = run_inference(request.text)
    log_prediction(request.text, label, confidence, latency_ms)
    return PredictResponse(label=label, confidence=confidence, latency_ms=latency_ms)

# Load the fine-tuned BERT model and tokenizer once at startup and expose a "/predict" endpoint that validates input with Pydantic, runs inference with gradient tracking off, and logs every prediction with its confidence and latency for future drift monitoring.
# I have a health endpoint for basic liveness checks.
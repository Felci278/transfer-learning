"""Urdu Sentiment Analysis — Inference Server

Four trained models:
  pt_full   Full fine-tune  (deployment_assets/full_ft_model)
  pt_lora   LoRA PEFT       (deployment_assets/lora_model)
  hf_full   Full fine-tune  (deployment_assets_hf/full_finetune_model)
  hf_lora   LoRA PEFT       (deployment_assets_hf/lora_model)

Endpoints:
  GET  /health         liveness check
  GET  /models         list loaded models and their label maps
  POST /predict        run one model  {"text": "...", "model": "hf_full"}
  POST /predict/all    run all models {"text": "..."}
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Disable network calls at runtime; base model is pre-cached during Docker build
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

PT_FULL_DIR  = BASE_DIR / "deployment_assets" / "full_ft_model"
PT_LORA_DIR  = BASE_DIR / "deployment_assets" / "lora_model"
HF_FULL_DIR  = BASE_DIR / "deployment_assets_hf" / "full_finetune_model"
HF_LORA_DIR  = BASE_DIR / "deployment_assets_hf" / "lora_model"
LABEL_MAPS   = BASE_DIR / "deployment_assets" / "label_maps.json"
HF_CACHE_DIR = BASE_DIR / "hf_cache"   # pre-populated during Docker build

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-multilingual-cased"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 128

# ── Model registry ────────────────────────────────────────────────────────────
MODELS: dict     = {}
TOKENIZERS: dict = {}


# ── Loaders ───────────────────────────────────────────────────────────────────
def _tokenizer(path: Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _full_model(model_dir: Path) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.eval().to(DEVICE)


def _lora_model(adapter_dir: Path, id2label: dict, label2id: dict) -> PeftModel:
    """Load a clean base model then graft the saved LoRA adapter on top.

    The base model is downloaded once during `docker build` into HF_CACHE_DIR
    so the container runs fully offline at inference time.
    """
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True)
    return model.eval().to(DEVICE)


def load_all_models() -> None:
    with open(LABEL_MAPS, encoding="utf-8") as fh:
        pt_maps = json.load(fh)

    pt_id2label  = pt_maps["id2label"]   # {"0": "Negative", "1": "Positive"}
    pt_label2id  = pt_maps["label2id"]   # {"Negative": 0, "Positive": 1}

    hf_id2label  = {"0": "negative", "1": "positive"}
    hf_label2id  = {"negative": 0, "positive": 1}

    # pt_full — PyTorch-loop full fine-tune
    TOKENIZERS["pt_full"] = _tokenizer(PT_FULL_DIR)
    MODELS["pt_full"]     = _full_model(PT_FULL_DIR)

    # pt_lora — PyTorch-loop LoRA adapter
    TOKENIZERS["pt_lora"] = _tokenizer(PT_FULL_DIR)
    MODELS["pt_lora"]     = _lora_model(PT_LORA_DIR, pt_id2label, pt_label2id)

    # hf_full — HF Trainer full fine-tune
    TOKENIZERS["hf_full"] = _tokenizer(HF_FULL_DIR)
    MODELS["hf_full"]     = _full_model(HF_FULL_DIR)

    # hf_lora — HF Trainer LoRA adapter
    TOKENIZERS["hf_lora"] = _tokenizer(HF_FULL_DIR)
    MODELS["hf_lora"]     = _lora_model(HF_LORA_DIR, hf_id2label, hf_label2id)

    print(f"[startup] {len(MODELS)} models loaded on {DEVICE}: {list(MODELS)}")


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    MODELS.clear()
    TOKENIZERS.clear()


app = FastAPI(
    title="Urdu Sentiment Analysis API",
    description=(
        "Binary sentiment classification (positive / negative) for Urdu and "
        "Roman-Urdu text. Serves four fine-tuned DistilBERT variants."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: str = "hf_full"


class PredictResponse(BaseModel):
    model: str
    text: str
    label: str
    score: float


class AllPredictResponse(BaseModel):
    text: str
    predictions: dict


# ── Inference helper ──────────────────────────────────────────────────────────
def _run_inference(model_key: str, text: str) -> dict:
    if model_key not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_key}'. Available: {list(MODELS)}",
        )

    tokenizer = TOKENIZERS[model_key]
    model     = MODELS[model_key]

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs   = torch.softmax(logits, dim=-1).squeeze()
    pred_id = int(torch.argmax(probs).item())

    # id2label keys may be int or str depending on model version
    cfg   = model.config
    label = cfg.id2label.get(pred_id) or cfg.id2label.get(str(pred_id), str(pred_id))
    score = round(float(probs[pred_id].item()), 4)

    return {"model": model_key, "text": text, "label": label, "score": score}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": list(MODELS)}


@app.get("/models")
def list_models():
    return {
        key: {
            "architecture": (model.config.architectures or ["unknown"])[0],
            "labels": model.config.id2label,
        }
        for key, model in MODELS.items()
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _run_inference(req.model, req.text)


@app.post("/predict/all", response_model=AllPredictResponse)
def predict_all(req: PredictRequest):
    predictions = {
        key: {"label": r["label"], "score": r["score"]}
        for key in MODELS
        for r in [_run_inference(key, req.text)]
    }
    return {"text": req.text, "predictions": predictions}

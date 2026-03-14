# Urdu Sentiment Analysis 

Binary sentiment classification (positive / negative) for Urdu and Roman-Urdu text using fine-tuned `distilbert-base-multilingual-cased`, served via FastAPI.

## Files

| File | Purpose |
|---|---|
| `Urdu_huggingface.ipynb` | Training — HF Trainer (Full FT + LoRA), runs on Apple MPS |
| `Urdu_pytorch.ipynb` | Training — PyTorch loop (Full FT + LoRA), runs on Apple MPS |
| `app.py` | FastAPI server — loads and serves all 4 trained models |
| `Dockerfile` | Containerises `app.py` with all model weights |
| `cache_model.py` | Downloads base model into image at build time |
| `requirements.txt` | Python dependencies (torch excluded, see below) |
| `deployment_assets/` | Model weights saved by `pytorch_1.ipynb` |
| `deployment_assets_hf/` | Model weights saved by `huggingface_1.ipynb` |

---


## Deployment

### Build and run

```bash
docker build -t urdu-sentiment:latest .
docker run -d --name urdu-sentiment -p 8000:8000 urdu-sentiment:latest
docker logs -f urdu-sentiment   # wait for "4 models loaded"
```

### Test

```bash
# Health
curl http://localhost:8000/health

# Single model (options: pt_full, pt_lora, hf_full, hf_lora)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Drama bohat acha tha", "model": "hf_full"}'

# Compare all 4 models
curl -X POST http://localhost:8000/predict/all \
  -H "Content-Type: application/json" \
  -d '{"text": "یہ پروڈکٹ بالکل بیکار ہے"}'
```

Swagger UI: `http://localhost:8000/docs`

### Stop / remove

```bash
docker stop urdu-sentiment
docker rm -f urdu-sentiment
```

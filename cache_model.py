"""Run once at Docker build time to pre-download the DistilBERT base model.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "distilbert-base-multilingual-cased"
CACHE = "/app/hf_cache"

print(f"Caching tokenizer for {MODEL} ...")
AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)

print(f"Caching model weights for {MODEL} ...")
AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2,
    cache_dir=CACHE,
)

print("Base model cached successfully.")

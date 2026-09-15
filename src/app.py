"""
src/app.py — FastAPI backend

Endpoints:
  GET  /                        → serves index.html
  POST /api/attention           → raw or rollout attention for one model
  POST /api/compare             → rollout side-by-side for BERT vs BioBERT
  POST /api/heads               → all layers + all heads for head explorer
  GET  /api/models              → list of supported models

Run:
  uvicorn src.app:app --reload --port 7860

For HuggingFace Spaces deployment:
  The Spaces runtime calls uvicorn automatically if
  app.py is at the root or you specify it in README.md YAML.
  Port must be 7860 for Spaces.
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from src.attention import AttentionExtractor, SUPPORTED_MODELS

app = FastAPI(title="Attention Visualiser")

# ── Serve static files ────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Model cache — load once, reuse across requests ────────────────────
_extractors = {}

def get_extractor(model_name: str) -> AttentionExtractor:
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model_name}"
        )
    if model_name not in _extractors:
        _extractors[model_name] = AttentionExtractor(model_name)
    return _extractors[model_name]


# ── Request / Response schemas ────────────────────────────────────────

class AttentionRequest(BaseModel):
    model_config = {"protected_namespaces": ()} 
    text:          str
    model_name:    str  = "bert-base-uncased"
    mode:          str  = "rollout"   # "raw" or "rollout"
    layer:         int  = -1          # only used for mode=raw
    head:          int  = 0           # only used for mode=raw
    discard_ratio: float = 0.1        # only used for mode=rollout

class CompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()} 
    text:           str
    model_names:    list = [
        "bert-base-uncased",
        "dmis-lab/biobert-base-cased-v1.2"
    ]
    discard_ratio:  float = 0.1

class HeadsRequest(BaseModel):
    model_config = {"protected_namespaces": ()} 
    text:       str
    model_name: str = "bert-base-uncased"


# ── Helpers ───────────────────────────────────────────────────────────

def normalise(arr: np.ndarray) -> list:
    """Normalise array to [0,1] and convert to Python list for JSON."""
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr.tolist()


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html",encoding="utf-8") as f:
        return f.read()


@app.get("/api/models")
async def list_models():
    return {
        name: info["display"]
        for name, info in SUPPORTED_MODELS.items()
    }


@app.post("/api/attention")
async def get_attention(req: AttentionRequest):
    """
    Returns attention matrix for the given text and model.
    mode=rollout → full-network rollout (recommended)
    mode=raw     → single layer + head (for comparison)
    """
    extractor = get_extractor(req.model_name)

    if req.mode == "rollout":
        result = extractor.get_rollout(
            req.text,
            discard_ratio=req.discard_ratio
        )
        return {
            "tokens":        result["tokens"],
            "attention":     normalise(result["attention"]),
            "cls_attention": normalise(result["cls_attention"]),
            "pool_token_idx":   result["pool_token_idx"],    # NEW
            "pool_token_label": result["pool_token_label"],  # NEW
            "mode":          "rollout",
            "n_layers":      result["n_layers"],
            "n_heads":       result["n_heads"],
            "model_display": SUPPORTED_MODELS[req.model_name]["display"],
        }

    elif req.mode == "raw":
        result = extractor.get_raw_attention(
            req.text, layer=req.layer, head=req.head
        )
        return {
            "tokens":    result["tokens"],
            "attention": normalise(result["attention"]),
            "layer":     result["layer"],
            "head":      result["head"],
            "pool_token_idx":   result["pool_token_idx"],    # NEW
            "pool_token_label": result["pool_token_label"],  # NEW
            "mode":      "raw",
            "n_layers":  result["n_layers"],
            "n_heads":   result["n_heads"],
            "model_display": SUPPORTED_MODELS[req.model_name]["display"],
        }

    else:
        raise HTTPException(status_code=400,
                            detail="mode must be 'rollout' or 'raw'")


@app.post("/api/compare")
async def compare_models(req: CompareRequest):
    """
    Side-by-side rollout comparison across two models.
    Used for BERT vs BioBERT view.
    """
    results = {}
    for name in req.model_names:
        extractor = get_extractor(name)
        r         = extractor.get_rollout(
            req.text, discard_ratio=req.discard_ratio
        )
        results[name] = {
            "display":       SUPPORTED_MODELS[name]["display"],
            "tokens":        r["tokens"],
            "cls_attention": normalise(r["cls_attention"]),
            "attention":     normalise(r["attention"]),
        }

    return {
        "text":    req.text,
        "results": results,
    }


@app.post("/api/heads")
async def explore_heads(req: HeadsRequest):
    """
    Returns all layers × all heads attention matrices.
    Used for the head-explorer view — lets you see
    what each head specialises in.
    """
    extractor = get_extractor(req.model_name)
    result    = extractor.get_all_layers_all_heads(req.text)

    # Shape: [n_layers, n_heads, seq, seq]
    attn = result["attentions"]

    # Normalise per head
    normalised = []
    for layer in range(result["n_layers"]):
        layer_data = []
        for head in range(result["n_heads"]):
            layer_data.append(normalise(attn[layer, head]))
        normalised.append(layer_data)

    return {
        "tokens":     result["tokens"],
        "attentions": normalised,
        "n_layers":   result["n_layers"],
        "n_heads":    result["n_heads"],
        "model_display": SUPPORTED_MODELS[req.model_name]["display"],
    }

# Add after existing endpoints

@app.post("/api/layerwise")
async def get_layerwise(req: AttentionRequest):
    extractor = get_extractor(req.model_name)
    result    = extractor.get_layerwise_cls_attention(req.text)
    return {
        "tokens":    result["tokens"],
        "layerwise": [normalise(np.array(layer)) 
                      for layer in result["layerwise"]],
        "pool_token_idx":   result["pool_token_idx"],    # NEW
        "pool_token_label": result["pool_token_label"],  # NEW
        "n_layers":  result["n_layers"],
    }


@app.post("/api/embeddings")
async def get_embeddings(req: AttentionRequest):
    extractor = get_extractor(req.model_name)
    result    = extractor.get_embedding_decomposition(req.text)
    return result


@app.post("/api/residual")
async def get_residual(req: AttentionRequest):
    extractor = get_extractor(req.model_name)
    result    = extractor.get_residual_contributions(req.text)
    return {
        "tokens":        result["tokens"],
        "contributions": [normalise(np.array(layer))
                         for layer in result["contributions"]],
        "n_layers":      result["n_layers"],
    }


@app.post("/api/entropy")
async def get_entropy(req: AttentionRequest):
    extractor = get_extractor(req.model_name)
    result    = extractor.get_head_entropy(req.text)
    return result


@app.post("/api/gradients")
async def get_gradients(req: AttentionRequest):
    extractor = get_extractor(req.model_name)
    result    = extractor.get_integrated_gradients(req.text)
    return result
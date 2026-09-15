# Transformer Interpretability: Attention Rollout and Attribution Across BERT and BioBERT

An interactive attention-interpretability tool for transformer text encoders,
built around attention **rollout** (Abnar & Zuidema, 2020) rather than raw
per-layer attention weights — the same limitation that makes tools like
BertViz show an incomplete picture of what a model is actually attending to.

**Status: core rollout engine, multi-view visualisation (heatmap, arc
diagram, head explorer, BERT vs. BioBERT comparison), and several
additional interpretability views implemented. Not yet deployed to
HuggingFace Spaces.**

---

## Why Rollout, Not Raw Attention

Most attention visualisers show raw attention weights from a single layer
and head. This is misleading: information flows through a transformer via
residual connections as well as attention, so a token that isn't directly
attended to in the final layer may still have shaped the output heavily
through earlier layers. Rollout accounts for this by propagating attention
through the full network depth — layer matrices are averaged over heads,
combined with the identity matrix (residual connection), row-normalized,
and multiplied together across all layers. The result is the *effective*
attention from input to output through the whole model, not just one slice
of it.

---

## What It Does

- **CLS Attention bars** — how much each token contributed to the final
  sentence representation, via rollout
- **Attention Heatmap** — full token-to-token attention matrix
- **Arc Diagram** — interactive per-token attention connections
- **BERT vs. BioBERT comparison** — same text, same rollout algorithm,
  different pretraining corpora (general vs. PubMed/PMC) — shows how
  domain pretraining changes what gets attended to
- **Head Explorer** — every attention head in every layer as a mini
  heatmap, click to enlarge
- **Layer Evolution** — how CLS attention shifts from early (surface
  patterns) to late (semantic) layers
- **Token vs. Position decomposition** — how much of each token's
  representation comes from its meaning vs. its position in the sequence
- **Residual Stream contributions** — how much each layer actually changes
  the representation vs. passing it through unchanged
- **Head Entropy** — which attention heads are focused (low entropy) vs.
  diffuse/near-inactive (high entropy)
- **Integrated Gradients** — true gradient-based token attribution
  (Sundararajan et al., 2017), as a check against attention-based
  explanations, which are not always reliable importance measures on
  their own

---

## Architecture

```
Browser (D3.js visualisation)
      │  fetch() calls
      ▼
┌─────────────────────┐
│  FastAPI backend      │  src/app.py
│  (src/app.py)          │  Model cache — loads each model once, reuses
└──────────┬────────────┘  across requests
           │
           ▼
┌─────────────────────┐
│  AttentionExtractor    │  src/attention.py
│                        │  Rollout, raw attention, head explorer,
│                        │  layerwise CLS, embedding decomposition,
│                        │  residual contributions, head entropy,
│                        │  integrated gradients
└──────────┬────────────┘
           │
           ▼
   BERT / BioBERT / CLIP (HuggingFace transformers, output_attentions=True)
```

---

## Models Supported

| Model | Type | Notes |
|---|---|---|
| `bert-base-uncased` | General English | Standard 12-layer BERT |
| `dmis-lab/biobert-base-cased-v1.2` | Biomedical | Pretrained on PubMed abstracts + PMC full text |
| `openai/clip-vit-base-patch32` | Multimodal (text encoder) | See limitation below — pooling differs from BERT |

---

### BERT vs BioBERT — Rollout Comparison
![Model Comparison](assets/bert_vs_biobert.png)
*Sentence: "Hepatosplenomegaly with elevated transaminase 
and hypoalbuminemia"*

## Known Limitations

- **CLIP's "CLS Attention" is not directly comparable to BERT/BioBERT's.**
  CLIP's text encoder pools its representation from the `<|endoftext|>`
  token position (found via argmax over token IDs), not position 0. The
  current implementation treats row 0 as "CLS attention" uniformly across
  all three models, which is accurate for BERT/BioBERT but not for CLIP.
  Fix: locate the actual EOS/pooling token position for CLIP rather than
  assuming index 0, or label the CLIP view differently to avoid implying
  equivalence.
- Not yet deployed to HuggingFace Spaces (port 7860 already configured for
  it, so deployment itself should be straightforward once ready).
- `compare_models()` reloads a second `AttentionExtractor` instance per
  comparison call rather than using the backend's model cache — works, but
  means BERT vs. BioBERT comparisons re-load BioBERT on every request
  rather than reusing a cached instance the way single-model requests do.
- Integrated Gradients uses a fixed 20-step Riemann approximation and a
  zero-vector baseline — reasonable defaults, but not validated against a
  reference IG implementation for numerical accuracy.

---

## Tech Stack

| Component | Tool |
|---|---|
| Backend | FastAPI |
| Models | BERT, BioBERT, CLIP (HuggingFace `transformers`) |
| Visualisation | D3.js v7 |
| Rollout algorithm | Abnar & Zuidema (2020) |
| Attribution (secondary) | Integrated Gradients (Sundararajan et al., 2017) |

---

## Running Locally

```bash
python main.py
# Serves at http://localhost:7860
```

## Deploying to HuggingFace Spaces

Port is already set to 7860 as required. Push this repo to a Spaces
repository with an `app.py`/`main.py` entry point and a `README.md` with
the Spaces YAML front matter (SDK: `docker` or `gradio`/`static`+FastAPI,
depending on your Space's runtime configuration).

---

## References

- [Quantifying Attention Flow in Transformers (Abnar & Zuidema, 2020)](https://arxiv.org/abs/2005.00928)
- [BertViz](https://github.com/jessevig/bertviz)
- [BioBERT: a pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365)

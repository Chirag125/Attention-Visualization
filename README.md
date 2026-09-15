# Transformer Interpretability: Attention Rollout and Attribution Across BERT and BioBERT

An interactive attention-interpretability tool for transformer text encoders,
built around attention **rollout** (Abnar & Zuidema, 2020) rather than raw
per-layer attention weights — the same limitation that makes tools like
BertViz show an incomplete picture of what a model is actually attending to.

**Status:** core rollout engine, multi-view visualisation (heatmap, arc
diagram, head explorer, BERT vs. BioBERT comparison), and several
additional interpretability views implemented. Not yet deployed to
HuggingFace Spaces.

---

## Why Rollout, Not Raw Attention

Most attention visualisers show raw attention weights from a single layer
and head. This is misleading: information flows through a transformer via
residual connections as well as attention, so a token that isn't directly
attended to in the final layer may still have shaped the output heavily
through earlier layers. Rollout accounts for this by propagating attention
through the full network depth — layer matrices are averaged over heads,
combined with the identity matrix (residual connection), row-normalised,
and multiplied together across all layers. The result is the *effective*
attention from input to output through the whole model, not just one slice
of it.

This matters particularly for biomedical text, where rare compound terms
like "hepatosplenomegaly" are fragmented into many subword tokens — raw
per-layer attention may highlight arbitrary fragments, while rollout reveals
which fragments collectively carry semantic weight through the full network.

---

## What It Does

- **CLS Attention bars** — how much each token contributed to the final
  sentence representation, via rollout. For CLIP, the pooling token is
  `<|endoftext|>` (EOS), not position 0 — the implementation detects this
  automatically and labels the view accordingly.
- **Attention Heatmap** — full token-to-token attention matrix
- **Arc Diagram** — interactive per-token attention connections; click any
  token to refocus
- **Layer Evolution** — how CLS attention shifts from early (surface
  patterns) to late (semantic) layers, with an animated playback slider
- **Token vs. Position decomposition** — how much of each token's
  layer-0 representation comes from its semantic embedding vs. its
  positional encoding
- **Residual Stream contributions** — how much each layer actually changes
  the representation (L2 norm of hidden state delta) vs. passing it
  through unchanged via the residual connection
- **Head Entropy** — which attention heads are focused (low entropy,
  tracking specific relationships) vs. diffuse/near-inactive (high entropy)
- **Integrated Gradients** — gradient-based token attribution
  (Sundararajan et al., 2017), as a check against attention-based
  explanations, which are not always reliable importance measures
- **BERT vs. BioBERT comparison** — same text, same rollout algorithm,
  different pretraining corpora (general vs. PubMed/PMC). Includes a
  token-difference row highlighting where the two models most strongly
  disagree.
- **Head Explorer** — every attention head in every layer as a mini
  heatmap; click any to enlarge in the Heatmap view

---

## Visualisations

### BERT vs BioBERT — Rollout Comparison

*Sentence: "Hepatosplenomegaly with elevated transaminase and hypoalbuminemia"*

<img width="1832" height="862" alt="BERT vs BioBERT comparison" src="https://github.com/user-attachments/assets/27f095fa-ff19-4136-8f06-4a2e9ee3d6a0" />

Both models use identical WordPiece tokenisation — the difference lies in
how attention weight distributes across the resulting subword fragments.
BERT anchors on word boundaries within the compound ("he" and "##y"),
treating it as an unfamiliar sequence to parse character by character.
BioBERT shows more uniform internal distribution across the subword
fragments and greater sensitivity to the relational conjunction "and"
connecting the two clinical findings — a difference consistent with
exposure to millions of PubMed abstracts where such conjunctions connect
distinct clinical entities. The orange token-difference row highlights
where the two models most strongly disagree.

### CLS / Pool Token Attention

*Sentence: "The patient shows signs of pneumothorax"*

<img width="1882" height="846" alt="CLS Attention" src="https://github.com/user-attachments/assets/e128696e-fd30-468f-825d-91ae1a91d394" />

Rollout attention (discard ratio 0.5) on BioBERT. Subword tokens are
shown individually — `##thor` and `##ax` are fragments of "pneumothorax".
Despite fragmentation, medical content tokens receive higher aggregate
attention than function words ("the", "shows", "of"), confirming that
rollout correctly propagates clinical salience through the full network
depth.

---

## Connection to Medical AI Portfolio

This tool exposes the attention mechanisms underlying the retrieval models
in the companion
[Medical Multimodal RAG](https://github.com/Chirag125/medical_rag) system.
BGE-large-en-v1.5 — the dense retrieval encoder in that pipeline — uses
the same transformer attention architecture as BERT. The interpretability
views here apply directly to understanding why certain medical text chunks
are retrieved over others.

The [Chest X-ray Detection](https://github.com/Chirag125/xray-detection)
system uses GradCAM for spatial interpretability — this tool provides the
NLP equivalent: token-level attribution for the text side of the clinical
AI pipeline.

---

## Architecture

```
Browser (D3.js visualisation)
      │  fetch() calls
      ▼
┌──────────────────────┐
│  FastAPI backend      │  src/app.py
│                       │  Model cache — loads each model once,
│                       │  reuses across requests
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│  AttentionExtractor   │  src/attention.py
│                       │  Rollout, raw attention, head explorer,
│                       │  layerwise CLS, embedding decomposition,
│                       │  residual contributions, head entropy,
│                       │  integrated gradients, pool token detection
└──────────┬────────────┘
           │
           ▼
   BERT / BioBERT / CLIP
   (HuggingFace transformers, output_attentions=True)
```

---

## Models Supported

| Model | Type | Notes |
|---|---|---|
| `bert-base-uncased` | General English | Standard 12-layer BERT |
| `dmis-lab/biobert-base-cased-v1.2` | Biomedical | Pretrained on PubMed abstracts + PMC full text |
| `openai/clip-vit-base-patch32` | Multimodal (text encoder) | Pools from `<|endoftext|>` (EOS), not position 0 — handled automatically |

---

## Known Limitations

- **Word-level aggregation not yet implemented.** Compound medical terms
  like "hepatosplenomegaly" appear as multiple subword bars rather than
  one. Planned: sum `##`-prefixed subword attention values back to their
  parent word before rendering, making comparisons more readable.

- **Integrated Gradients uses a 20-step Riemann approximation with a
  zero-vector baseline.** Reasonable defaults, but not validated against
  a reference IG implementation. Results should be treated as directional
  rather than numerically precise.

- **Discard ratio requires manual tuning.** Default 0.5 works well for
  most sentences; shorter or simpler sentences may need lower values
  to avoid over-suppression.

- **Not yet deployed to HuggingFace Spaces.** Port 7860 is already
  configured. Deployment is straightforward once ready.

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
pip install -r requirements.txt
python main.py
# Serves at http://localhost:7860
```

## Deploying to HuggingFace Spaces

Port is already set to 7860. Push to a Spaces repository with a
`Dockerfile` and `README.md` with the Spaces YAML front matter
(`sdk: docker`, `app_port: 7860`).

---

## References

- [Quantifying Attention Flow in Transformers — Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928)
- [BertViz — Jesse Vig](https://github.com/jessevig/bertviz)
- [BioBERT: a pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Axiomatic Attribution for Deep Networks — Sundararajan et al. (2017)](https://arxiv.org/abs/1703.01365)
- [What Does BERT Look at? — Clark et al. (2019)](https://arxiv.org/abs/1906.04341)
- [Medical Multimodal RAG (companion project)](https://github.com/Chirag125/medical_rag)
- [Chest X-ray Detection (companion project)](https://github.com/Chirag125/xray-detection)

"""
src/attention.py — Attention extraction with rollout

Three models supported:
  BERT        — general English text (bert-base-uncased)
  BioBERT     — biomedical text (dmis-lab/biobert-base-cased-v1.2)
  CLIP (text) — multimodal encoder (openai/clip-vit-base-patch32)

Two attention modes:
  raw      — attention weights from one layer/head (standard, misleading)
  rollout  — Abnar & Zuidema 2020 — multiplies attention across all layers
             accounting for residual connections
             This is what makes this visualiser different from BertViz

Why rollout matters:
  Raw attention weights from layer 12 show only that layer's focus.
  Information flows through ALL layers via residual connections.
  A token that isn't directly attended to in layer 12 may still
  have contributed heavily through earlier layers.
  Rollout computes the effective attention from input to output
  through the full depth of the network.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPTokenizer,
    CLIPTextModel,
)


# ── Model registry ────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "bert-base-uncased": {
        "display": "BERT (General)",
        "type":    "bert",
    },
    "dmis-lab/biobert-base-cased-v1.2": {
        "display": "BioBERT (Biomedical)",
        "type":    "bert",
    },
    "openai/clip-vit-base-patch32": {
        "display": "CLIP Text Encoder",
        "type":    "clip",
    },
}


# ── Attention extractor ───────────────────────────────────────────────

class AttentionExtractor:
    """
    Extracts attention weights from BERT-family or CLIP text encoders.
    Supports both raw per-layer attention and rollout aggregation.
    """

    def __init__(self, model_name="bert-base-uncased"):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from: {list(SUPPORTED_MODELS)}"
            )

        self.model_name  = model_name
        self.model_info  = SUPPORTED_MODELS[model_name]
        self.model_type  = self.model_info["type"]
        self.device      = torch.device("cpu")

        print(f"Loading {self.model_info['display']}...")
        if self.model_type == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model     = CLIPTextModel.from_pretrained(
                model_name, output_attentions=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model     = AutoModel.from_pretrained(
                model_name, output_attentions=True
            )

        self.model.to(self.device)
        self.model.eval()
        print(f"  Ready — {self.count_layers()} layers, "
              f"{self.count_heads()} heads/layer\n")

    def count_layers(self):
        try:
            return self.model.config.num_hidden_layers
        except AttributeError:
            return self.model.config.text_config.num_hidden_layers

    def count_heads(self):
        try:
            return self.model.config.num_attention_heads
        except AttributeError:
            return self.model.config.text_config.num_attention_heads

    def tokenize(self, text, max_length=64):
        """Tokenise input and return tokens + ids."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0]
        )
        return encoding, tokens

    def get_raw_attention(self, text, layer=-1, head=0):
        """
        Returns raw attention weights from one layer and one head.
        Shape: [seq_len, seq_len]

        layer: -1 = last layer (default)
        head:   0 = first head (default)

        This is what most attention visualisers show — but it's
        an incomplete picture because it ignores residual connections
        and only shows one layer's focus.
        """
        encoding, tokens = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**encoding)

        # outputs.attentions: tuple of [1, n_heads, seq, seq] per layer
        attentions = outputs.attentions
        n_layers   = len(attentions)

        layer_idx = layer % n_layers  # handle negative indexing
        attn      = attentions[layer_idx][0, head]  # [seq, seq]
        attn_np   = attn.cpu().numpy()

        return {
            "tokens":    tokens,
            "attention": attn_np,
            "layer":     layer_idx,
            "head":      head,
            "n_layers":  n_layers,
            "n_heads":   attentions[0].shape[1],
            "mode":      "raw",
        }

    def get_rollout(self, text, discard_ratio=0.0):
        """
        Attention rollout — Abnar & Zuidema 2020.

        Algorithm:
          1. For each layer, average attention over all heads
          2. Add identity matrix (residual connection)
          3. Normalise rows to sum to 1
          4. Multiply all layer matrices together (left to right)

        Result: effective attention from each input token to
                each output token through the FULL network.

        discard_ratio: fraction of lowest attention weights to zero out
                       before rollout (reduces noise). 0.0 = no discard.

        Returns rollout matrix [seq_len, seq_len].
        [CLS] → token attention in row 0 tells you which tokens
        contributed most to the final [CLS] representation.
        """
        encoding, tokens = self.tokenize(text)
        seq_len          = encoding["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**encoding)

        attentions = outputs.attentions  # tuple of [1, heads, seq, seq]

        # Step 1: average heads per layer
        # attn_layers: list of [seq, seq] arrays
        attn_layers = []
        for layer_attn in attentions:
            avg = layer_attn[0].mean(dim=0).cpu().numpy()  # [seq, seq]

            # Optional: discard lowest attention weights
            if discard_ratio > 0:
                flat       = avg.flatten()
                threshold  = np.quantile(flat, discard_ratio)
                avg        = np.where(avg > threshold, avg, 0.0)

            attn_layers.append(avg)

        # Step 2: add residual connection (identity matrix) and normalise
        eye = np.eye(seq_len)
        processed = []
        for attn in attn_layers:
            a = attn + eye
            # Row-normalise: each row sums to 1
            row_sums = a.sum(axis=-1, keepdims=True)
            a        = a / (row_sums + 1e-8)
            processed.append(a)

        # Step 3: matrix product across all layers
        rollout = processed[0]
        for i in range(1, len(processed)):
            rollout = processed[i] @ rollout

        return {
            "tokens":         tokens,
            "attention":      rollout,
            "cls_attention":  rollout[0],  # [CLS] → all tokens
            "n_layers":       len(attentions),
            "n_heads":        attentions[0].shape[1],
            "mode":           "rollout",
            "discard_ratio":  discard_ratio,
        }

    def get_all_layers_all_heads(self, text):
        """
        Returns raw attention for ALL layers and ALL heads.
        Used for the head-explorer view in the UI.
        Shape: [n_layers, n_heads, seq_len, seq_len]
        """
        encoding, tokens = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**encoding)

        attentions = outputs.attentions
        # Stack: [n_layers, 1, n_heads, seq, seq] → squeeze batch
        all_attn = torch.stack([a[0] for a in attentions])
        # [n_layers, n_heads, seq, seq]

        return {
            "tokens":     tokens,
            "attentions": all_attn.cpu().numpy(),
            "n_layers":   len(attentions),
            "n_heads":    attentions[0].shape[1],
        }

    def compare_models(self, text, model_names=None):
        """
        Extract rollout from multiple models and return side-by-side.
        Used for BERT vs BioBERT comparison view.
        """
        if model_names is None:
            model_names = [
                "bert-base-uncased",
                "dmis-lab/biobert-base-cased-v1.2",
            ]

        results = {}
        current = self.model_name

        for name in model_names:
            if name != current:
                # Temporarily load the other model
                other = AttentionExtractor(name)
                r     = other.get_rollout(text)
            else:
                r     = self.get_rollout(text)

            results[name] = {
                "display":       SUPPORTED_MODELS[name]["display"],
                "tokens":        r["tokens"],
                "cls_attention": r["cls_attention"],
                "rollout":       r["attention"],
            }

        return results
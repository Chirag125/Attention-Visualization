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

    def get_pool_token_idx(self, tokens):
        """
        Returns the index of the token whose representation
        is used as the sentence-level embedding.

        BERT / BioBERT: position 0 ([CLS] token)
        CLIP:           position of <|endoftext|> (EOS token)
                        CLIP pools from the EOS position, not position 0.
                        Found via argmax of input_ids matching eos_token_id.
        """
        if self.model_type == "clip":
            # CLIP's EOS token is '49407' for openai/clip-vit-base-patch32
            # It appears as '<|endoftext|>' in the token list
            eos_candidates = [
                i for i, t in enumerate(tokens)
                if t in ('<|endoftext|>', '</s>', '[SEP]')
            ]
            return eos_candidates[-1] if eos_candidates else len(tokens) - 1
        else:
            # BERT and BioBERT always use position 0 ([CLS])
            return 0

    def get_pool_token_label(self):
        """
        Human-readable label for the pooling token.
        Used in the UI so the label matches the actual token.
        """
        if self.model_type == "clip":
            return "<|endoftext|> (EOS)"
        return "[CLS]"
    
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

        pool_idx = self.get_pool_token_idx(tokens)

        return {
            "tokens":           tokens,
            "attention":        attn_np,
            "cls_attention":    attn_np[pool_idx],  # was always attn_np[0]
            "pool_token_idx":   pool_idx,
            "pool_token_label": self.get_pool_token_label(),
            "layer":            layer_idx,
            "head":             head,
            "n_layers":         n_layers,
            "n_heads":          attentions[0].shape[1],
            "mode":             "raw",
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

        pool_idx = self.get_pool_token_idx(tokens)

        special = {'[CLS]', '[SEP]', '<s>', '</s>',
           '<|startoftext|>', '<|endoftext|>'}
        cls_attn = rollout[pool_idx].copy()
        for i, tok in enumerate(tokens):
            if tok in special:
                cls_attn[i] = 0.0

        return {
            "tokens":           tokens,
            "attention":        rollout,
            "cls_attention":    cls_attn,    # special tokens zeroed
            "pool_token_idx":   pool_idx,
            "pool_token_label": self.get_pool_token_label(),
            "n_layers":         len(attentions),
            "n_heads":          attentions[0].shape[1],
            "mode":             "rollout",
            "discard_ratio":    discard_ratio,
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
    # Add to src/attention.py

    def get_layerwise_cls_attention(self, text):
        encoding, tokens = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**encoding)

        attentions = outputs.attentions
        pool_idx   = self.get_pool_token_idx(tokens)

        # Identify special tokens to exclude from normalisation
        special = {'[CLS]', '[SEP]', '<s>', '</s>',
                '<|startoftext|>', '<|endoftext|>'}

        layerwise = []
        for layer_attn in attentions:
            avg_attn  = layer_attn[0].mean(dim=0)
            pool_attn = avg_attn[pool_idx].cpu().numpy()

            # Zero out special tokens so they don't dominate normalisation
            masked = pool_attn.copy()
            for i, tok in enumerate(tokens):
                if tok in special:
                    masked[i] = 0.0

            layerwise.append(masked.tolist())

        return {
            "tokens":           tokens,
            "layerwise":        layerwise,
            "pool_token_idx":   pool_idx,
            "pool_token_label": self.get_pool_token_label(),
            "n_layers":         len(layerwise),
        }

    def get_embedding_decomposition(self, text):
        """
        Decomposes layer-0 input into:
        - Token embedding contribution (what the word means)
        - Positional encoding contribution (where the word is)
        
        Method: extract embeddings and positional encodings separately,
        compute their L2 norms, report ratio.
        
        This shows: which tokens are identified more by meaning vs position?
        Clinical insight: rare medical terms rely heavily on semantic embedding,
        common words like 'with' and 'the' are more position-dependent.
        """
        encoding, tokens = self.tokenize(text)
        
        if self.model_type == "clip":
            emb_layer = self.model.text_model.embeddings
        else:
            emb_layer = self.model.embeddings
        
        input_ids      = encoding["input_ids"]
        position_ids   = torch.arange(input_ids.shape[1]).unsqueeze(0)
        
        with torch.no_grad():
            # Token embeddings only (no position)
            token_embeds = emb_layer.word_embeddings(input_ids)
            
            # Positional encodings only
            pos_embeds   = emb_layer.position_embeddings(position_ids)
        
        # L2 norm of each token's semantic vs positional component
        token_norms = token_embeds[0].norm(dim=-1).cpu().numpy()
        pos_norms   = pos_embeds[0].norm(dim=-1).cpu().numpy()
        
        # Ratio: how much of this token's representation is positional?
        total = token_norms + pos_norms + 1e-8
        pos_ratio   = (pos_norms / total).tolist()
        token_ratio = (token_norms / total).tolist()
        
        return {
            "tokens":      tokens,
            "token_ratio": token_ratio,   # semantic embedding contribution
            "pos_ratio":   pos_ratio,     # positional encoding contribution
            "token_norms": token_norms.tolist(),
            "pos_norms":   pos_norms.tolist(),
        }
    # Add to src/attention.py

    def get_residual_contributions(self, text):
        """
        Tracks how much each transformer layer CHANGES the representation
        versus passing it through unchanged.
        
        High change at a layer = that layer is doing important processing.
        Low change = residual connection dominates, layer is a pass-through.
        
        Returns per-layer, per-token contribution magnitude.
        """
        encoding, tokens = self.tokenize(text)
        
        # Register hooks to capture hidden states at each layer
        hidden_states = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0].detach())
            else:
                hidden_states.append(output.detach())
        
        # Hook each transformer layer
        hooks = []
        if self.model_type == "clip":
            layers = self.model.text_model.encoder.layers
        else:
            layers = self.model.encoder.layer
        
        for layer in layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)
        
        for h in hooks:
            h.remove()
        
        # Get all hidden states including embedding layer
        all_hidden = outputs.hidden_states  # tuple: [n_layers+1, batch, seq, dim]
        
        contributions = []
        for i in range(1, len(all_hidden)):
            prev  = all_hidden[i-1][0]   # [seq, dim]
            curr  = all_hidden[i][0]     # [seq, dim]
            delta = (curr - prev).norm(dim=-1)  # [seq] — how much changed
            contributions.append(delta.cpu().numpy().tolist())
        
        return {
            "tokens":        tokens,
            "contributions": contributions,   # [n_layers, seq_len]
            "n_layers":      len(contributions),
        }
    # Add to src/attention.py

    def get_head_entropy(self, text):
        """
        Computes Shannon entropy of each attention head's distribution.
        
        Low entropy  = focused head (tracks specific relationships)
        High entropy = diffuse head (uniform attention = not doing much)
        
        Useful for identifying which heads are actually doing work
        versus which are effectively inactive.
        """
        encoding, tokens = self.tokenize(text)
        seq_len = encoding["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        attentions = outputs.attentions  # [n_layers × (1, heads, seq, seq)]
        
        entropies = []
        for layer_attn in attentions:
            layer_ent = []
            for head in range(layer_attn.shape[1]):
                attn  = layer_attn[0, head]  # [seq, seq]
                # Entropy of each row (query token's distribution)
                # H = -sum(p * log(p))
                attn_safe = attn.clamp(min=1e-9)
                ent = -(attn_safe * attn_safe.log()).sum(dim=-1)  # [seq]
                layer_ent.append(ent.mean().item())  # scalar per head
            entropies.append(layer_ent)
        
        return {
            "tokens":     tokens,
            "entropies":  entropies,   # [n_layers, n_heads]
            "n_layers":   len(entropies),
            "n_heads":    len(entropies[0]),
        }
    # Add to src/attention.py

    def get_integrated_gradients(self, text, target_token_idx=0, steps=20):
        """
        Integrated gradients: true token importance via gradient attribution.
        
        Unlike attention weights, this measures how much changing each
        input token would actually affect the output representation.
        
        target_token_idx: which output token to explain (0 = [CLS])
        steps: number of interpolation steps (more = more accurate, slower)
        
        Reference: Sundararajan et al. 2017 — "Axiomatic Attribution for DNNs"
        """
        encoding, tokens = self.tokenize(text)
        
        if self.model_type == "clip":
            emb_layer = self.model.text_model.embeddings.word_embeddings
        else:
            emb_layer = self.model.embeddings.word_embeddings
        
        input_ids    = encoding["input_ids"]
        embeds       = emb_layer(input_ids)          # [1, seq, dim]
        baseline     = torch.zeros_like(embeds)      # zero baseline
        
        integrated = torch.zeros_like(embeds)
        
        for step in range(steps + 1):
            alpha        = step / steps
            interp       = baseline + alpha * (embeds - baseline)
            interp       = interp.detach().requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            # Need to pass through model using inputs_embeds
            outputs = self.model(
                input_ids=None,
                inputs_embeds=interp,
                attention_mask=encoding.get("attention_mask")
            )
            
            # Target: L2 norm of target token's hidden state
            target = outputs.last_hidden_state[0, target_token_idx].norm()
            target.backward()
            
            integrated += interp.grad.detach()
        
        # IG attribution = (original - baseline) * avg_gradient
        attribution = ((embeds - baseline) * integrated / (steps + 1))
        # Sum over embedding dimension → scalar per token
        importance = attribution[0].norm(dim=-1).cpu().numpy()
        
        # Normalise
        importance = (importance - importance.min()) / \
                    (importance.max() - importance.min() + 1e-8)
        
        return {
            "tokens":     tokens,
            "importance": importance.tolist(),
            "target_idx": target_token_idx,
            "method":     "integrated_gradients",
        }
    def aggregate_to_words(self, tokens, attention_values):
        """
        Aggregates subword token attention back to whole words.
        
        BERT tokenises "Hepatosplenomegaly" as:
        ["He", "##pa", "##tos", "##ple", "##no", "##me", "##gal", "##y"]
        
        This sums their attention values into one bar for the whole word.
        Makes the visualisation readable — you see "Hepatosplenomegaly" 
        as one unit, not eight fragments.
        
        Returns: (word_list, aggregated_values)
        """
        words  = []
        values = []
        current_word  = ""
        current_value = 0.0

        for tok, val in zip(tokens, attention_values):
            # Skip special tokens
            if tok in ('[CLS]', '[SEP]', '<s>', '</s>',
                    '<|startoftext|>', '<|endoftext|>'):
                continue

            if tok.startswith("##"):
                # Continuation of previous word
                current_word  += tok[2:]
                current_value += val
            else:
                # Save previous word if exists
                if current_word:
                    words.append(current_word)
                    values.append(current_value)
                # Start new word
                current_word  = tok
                current_value = val

        # Save last word
        if current_word:
            words.append(current_word)
            values.append(current_value)

        return words, values

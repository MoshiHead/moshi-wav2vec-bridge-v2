"""
model.py — Mimi-to-Wav2Vec2 Bridge Module
==========================================
Converts Mimi discrete token streams (B, T, 8) → Wav2Vec2-like features.

Architecture mirrors facebook/wav2vec2-base-960h so that AvatarForcing can
replace its Wav2Vec encoder with this module with zero code changes:

  AvatarForcing consumption pattern (dataset.py):
    hs = bridge(tokens, seq_len=T, output_hidden_states=True)
    audio_emb = hs.last_hidden_state                    # (B, T, 768)
    for h in hs.hidden_states:                          # 13 tensors
        audio_emb = torch.cat([audio_emb, h], dim=-1)
    # → audio_emb: (B, T, 768 × 14) = (B, T, 10752)

Output rate: Mimi 12.5 Hz × upsample_factor 2 = 25 Hz  (matches AvatarForcing fps=25)

Transformer depth: 12 layers  (matches wav2vec2-base)
d_model          : 768         (matches wav2vec2-base hidden size)
nhead            : 12          (768 / 12 = 64 head_dim)
dim_feedforward  : 3072        (matches wav2vec2-base 4 × 768)

hidden_states tuple (13 items):
  [0]  = output of CausalUpsample + input_proj (pre-encoder, analogous to
          wav2vec2's feature_projection output before the transformer)
  [1..12] = output after each of the 12 CausalTransformer layers
last_hidden_state = hidden_states[12]
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Output container  (mirrors transformers.modeling_outputs.BaseModelOutput)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Wav2Vec2LikeOutput:
    """
    Drop-in replacement for transformers.BaseModelOutput.

    Attributes
    ----------
    last_hidden_state : (B, T, 768)
        Output of the final transformer layer.
    hidden_states : tuple of 13 × (B, T, 768)
        [0]  = pre-transformer projected features  (feature_projection analogue)
        [1]  = output of transformer layer 1
        ...
        [12] = output of transformer layer 12  (== last_hidden_state)
    """
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encodings
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias added to attention logits.
    Supports causal (unidirectional) masking.
    """

    def __init__(self, num_heads: int, max_distance: int = 128, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.causal = causal
        self.embeddings = nn.Embedding(2 * max_distance + 1, num_heads)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns (num_heads, T, T) bias tensor."""
        pos = torch.arange(seq_len, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.embeddings(rel)  # (T, T, H)
        bias = bias.permute(2, 0, 1)  # (H, T, T)
        if self.causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            bias = bias.masked_fill(mask.unsqueeze(0), float("-inf"))
        return bias


# ──────────────────────────────────────────────────────────────────────────────
# Causal Multi-Head Attention with Optional Relative Bias
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_relative_pe: bool = True,
        max_distance: int = 128,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.use_relative_pe = use_relative_pe
        if use_relative_pe:
            self.rel_bias = RelativePositionBias(nhead, max_distance, causal=True)

        # KV-cache (used during streaming inference)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        self._cache_k = None
        self._cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None
        S = k.size(2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, S)

        if self.use_relative_pe:
            bias = self.rel_bias(S, x.device)  # (H, S, S)
            attn = attn + bias[:, -T:, :]
        else:
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device), diagonal=S - T + 1
            ).bool()
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out, present_kv


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Layer & Stack
# ──────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):

    def __init__(
        self, d_model: int, nhead: int, dim_ff: int, dropout: float, use_relative_pe: bool
    ):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead, dropout, use_relative_pe)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv=None,
    ):
        attn_out, present_kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, present_kv


class CausalTransformer(nn.Module):
    """
    12-layer causal transformer that collects all intermediate hidden states,
    mirroring the wav2vec2-base encoder's output_hidden_states=True behaviour.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        use_relative_pe: bool,
        max_seq_len: int,
    ):
        super().__init__()
        self.use_relative_pe = use_relative_pe
        if not use_relative_pe:
            self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, dim_ff, dropout, use_relative_pe)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kvs=None,
        output_hidden_states: bool = False,
    ):
        """
        Returns
        -------
        x            : final hidden state  (B, T, d_model)
        present_kvs  : list of (k, v) per layer (only when use_cache=True)
        all_hidden   : tuple of (B, T, d_model) per layer (when output_hidden_states=True)
                       Does NOT include the pre-transformer input here; that is
                       added by MimiWav2Vec2Bridge.forward() as hidden_states[0].
        """
        if not self.use_relative_pe:
            x = self.pos_enc(x)

        present_kvs = []
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, pres = layer(x, use_cache=use_cache, past_kv=pkv)
            present_kvs.append(pres)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.norm(x)
        # Replace last entry with post-norm version (consistent with how
        # wav2vec2's encoder returns the final normalised state)
        if output_hidden_states:
            all_hidden[-1] = x

        return (
            x,
            present_kvs if use_cache else None,
            tuple(all_hidden) if output_hidden_states else None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Codebook Embedding
# ──────────────────────────────────────────────────────────────────────────────

class MultiCodebookEmbedding(nn.Module):
    """
    8 separate embedding tables (one per codebook level).
    Fusion: element-wise sum → (B, T, embed_dim)
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        vocab_size: int = 2048,
        embed_dim: int = 256,
        fusion: str = "sum",
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.fusion = fusion

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)]
        )

        if fusion == "concat":
            self.proj = nn.Linear(embed_dim * num_codebooks, embed_dim)

        self.level_scale = nn.Parameter(torch.ones(num_codebooks))
        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, num_codebooks)  integer indices
        returns: (B, T, embed_dim)
        """
        embeds = []
        for i, emb in enumerate(self.embeddings):
            e = emb(tokens[:, :, i])  # (B, T, embed_dim)
            e = e * self.level_scale[i]
            embeds.append(e)

        if self.fusion == "sum":
            out = torch.stack(embeds, dim=0).sum(dim=0)
        elif self.fusion == "concat":
            out = torch.cat(embeds, dim=-1)
            out = self.proj(out)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Causal Temporal Upsampler (×2)
# ──────────────────────────────────────────────────────────────────────────────

class CausalUpsample(nn.Module):
    """
    ConvTranspose1d (×upsample_factor) followed by a causal conv.
    Input:  (B, D, T)
    Output: (B, D, upsample_factor × T)
    Default factor=2: Mimi 12.5 Hz → 25 Hz  (matches AvatarForcing fps=25)
    """

    def __init__(self, channels: int, upsample_factor: int = 2):
        super().__init__()
        self.factor = upsample_factor

        self.conv_t = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
        )

        # Causal refinement conv (kernel=3, causal padding)
        self.causal_refine = nn.Conv1d(channels, channels, kernel_size=3, padding=0)
        self.causal_pad = 2  # (kernel - 1)

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T)"""
        x = self.conv_t(x)                      # (B, D, factor*T)
        x_pad = F.pad(x, (self.causal_pad, 0))
        x = self.causal_refine(x_pad)           # (B, D, factor*T)
        x = self.act(self.norm(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main Bridge Module
# ──────────────────────────────────────────────────────────────────────────────

class MimiWav2Vec2Bridge(nn.Module):
    """
    Full Mimi → Wav2Vec2-compatible bridge.

    Input  : tokens  (B, T_mimi, 8)   — Mimi discrete indices at 12.5 Hz
    Output : Wav2Vec2LikeOutput
        .last_hidden_state  (B, 2*T_mimi, 768)   at 25 Hz
        .hidden_states      tuple of 13 × (B, 2*T_mimi, 768)

    AvatarForcing drop-in usage (mirrors dataset.py exactly):
        hs = bridge(tokens, output_hidden_states=True)
        audio_emb = hs.last_hidden_state
        for h in hs.hidden_states:
            audio_emb = torch.cat([audio_emb, h], dim=-1)
        # → (B, 2*T_mimi, 768 × 14) = (B, 2*T_mimi, 10752)

    The 13 hidden_states are:
        [0]  pre-transformer features  (post upsample + input_proj, pre layer 1)
        [1]  output of transformer layer 1
        ...
        [12] output of transformer layer 12  (== last_hidden_state)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]

        self.embed_dim = m["embed_dim"]
        self.d_model = m["d_model"]          # must equal output_dim = 768
        self.output_dim = m["output_dim"]    # 768
        self.upsample_factor = m["upsample_factor"]

        assert self.d_model == self.output_dim, (
            f"d_model ({self.d_model}) must equal output_dim ({self.output_dim}) "
            "so that all hidden states are 768-dim without extra projections."
        )

        # 1. Multi-codebook embedding
        self.embedding = MultiCodebookEmbedding(
            num_codebooks=m["num_codebooks"],
            vocab_size=m["vocab_size"],
            embed_dim=m["embed_dim"],
            fusion=m["embed_fusion"],
        )

        # 2. Input projection embed_dim → d_model
        if m["embed_dim"] != m["d_model"]:
            self.input_proj = nn.Linear(m["embed_dim"], m["d_model"])
        else:
            self.input_proj = nn.Identity()

        # 3. Temporal upsampling ×2  (12.5 Hz → 25 Hz)
        self.upsample = CausalUpsample(m["d_model"], m["upsample_factor"])

        # 4. Pre-encoder LayerNorm  (analogous to wav2vec2's feature_projection norm)
        self.pre_norm = nn.LayerNorm(m["d_model"])

        # 5. 12-layer Causal Transformer  (matches wav2vec2-base depth)
        use_rel = m["pos_encoding"] == "relative"
        self.transformer = CausalTransformer(
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],   # 12
            dim_ff=m["dim_feedforward"],
            dropout=m["dropout"],
            use_relative_pe=use_rel,
            max_seq_len=m["max_seq_len"],
        )

        # Note: no output_proj needed — d_model == output_dim == 768.
        # Adding a projection here would create unused parameters in DDP
        # (only last_hidden_state participates in the MSE loss, and it is
        # already 768-dim straight from the transformer).

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        past_kvs=None,
    ) -> Tuple["Wav2Vec2LikeOutput", Optional[list]]:
        """
        Parameters
        ----------
        tokens            : (B, T, 8)  — integer Mimi indices
        output_hidden_states : if True, populate .hidden_states (13 tensors)
        use_cache         : enable KV-cache for streaming inference
        past_kvs          : cached (k, v) pairs from previous step

        Returns
        -------
        output     : Wav2Vec2LikeOutput
        present_kvs: list of (k,v) per layer (None when use_cache=False)
        """
        # ── Embedding ────────────────────────────────────────────────────────
        x = self.embedding(tokens)       # (B, T, embed_dim)
        x = self.input_proj(x)           # (B, T, d_model=768)

        # ── Temporal upsample  ×2  →  25 Hz ──────────────────────────────────
        x = x.transpose(1, 2)            # (B, d_model, T)
        x = self.upsample(x)             # (B, d_model, 2T)
        x = x.transpose(1, 2)           # (B, 2T, d_model)

        # ── Pre-encoder norm  (wav2vec2 feature_projection analogue) ─────────
        x = self.pre_norm(x)             # (B, 2T, 768)

        # This is hidden_states[0] — the "pre-encoder" representation
        h0 = x

        # ── 12-layer Causal Transformer ───────────────────────────────────────
        x, present_kvs, layer_hidden_states = self.transformer(
            x,
            use_cache=use_cache,
            past_kvs=past_kvs,
            output_hidden_states=output_hidden_states,
        )
        # x == last_hidden_state  (B, 2T, 768)

        # ── Build output object ───────────────────────────────────────────────
        # x is already 768-dim (d_model == output_dim); no projection needed.
        # last_hidden_state == hidden_states[-1] — matches wav2vec2 behaviour.
        if output_hidden_states:
            # 13 tensors: h0 (pre-encoder) + outputs of layers 1..12
            all_hidden = (h0,) + layer_hidden_states  # len == 13
        else:
            all_hidden = None

        output = Wav2Vec2LikeOutput(
            last_hidden_state=x,       # (B, 2T, 768)
            hidden_states=all_hidden,  # 13 × (B, 2T, 768) or None
        )

        return output, present_kvs

    # ------------------------------------------------------------------
    # Convenience: replicate AvatarForcing concatenation in one call
    # ------------------------------------------------------------------
    def encode_for_avatarforcing(
        self,
        tokens: torch.Tensor,
        use_cache: bool = False,
        past_kvs=None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Returns the 10752-dim concatenated tensor expected by AvatarForcing's
        diffusion model, plus the KV-cache.

        output shape: (B, 2T, 10752)  — ready to use as `audio_emb`
        """
        hs, present_kvs = self.forward(
            tokens, output_hidden_states=True, use_cache=use_cache, past_kvs=past_kvs
        )
        audio_emb = hs.last_hidden_state
        for h in hs.hidden_states:
            audio_emb = torch.cat([audio_emb, h], dim=-1)
        # audio_emb: (B, 2T, 768 × 14) = (B, 2T, 10752)
        return audio_emb, present_kvs

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# Backward-compatibility alias
MimiHuBERTBridge = MimiWav2Vec2Bridge


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator (for adversarial loss)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureDiscriminator(nn.Module):
    """
    Multi-scale 1D convolutional discriminator.
    Input: feature sequence (B, T, output_dim)
    Output: scalar real/fake logits
    """

    def __init__(self, input_dim: int = 768, hidden: int = 512, num_layers: int = 4):
        super().__init__()
        layers = []
        in_ch = input_dim
        for _ in range(num_layers):
            out_ch = hidden
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
            ]
            in_ch = out_ch
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x.transpose(1, 2)  # (B, D, T)
        return self.net(x)      # (B, 1, T')

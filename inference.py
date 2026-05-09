"""
inference.py — Streaming & Batch Inference
==========================================
Provides:
  - BridgeInference: batch mode
  - StreamingBridgeInference: causal chunk-by-chunk streaming
  - CLI entry point  (including --compare mode)

Both modes produce Wav2Vec2-compatible features (25 Hz, 768-dim per layer)
from Mimi tokens.

Output formats
--------------
  DEFAULT  → (B, 2T, 768)   last_hidden_state only  (for training/comparison)
  AVATARFORCING → (B, 2T, 10752)  14-layer concat   (drop-in for AvatarForcing)

AvatarForcing concat (mirrors dataset.py exactly):
  audio_emb = hs.last_hidden_state              # (B, T, 768)
  for h in hs.hidden_states:                    # 13 tensors
      audio_emb = torch.cat([audio_emb, h], -1)
  # → (B, T, 768 × 14) = (B, T, 10752)
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

from model import MimiHuBERTBridge, Wav2Vec2LikeOutput

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: replicate AvatarForcing's audio_emb concatenation
# ──────────────────────────────────────────────────────────────────────────────

def avatarforcing_concat(hs: Wav2Vec2LikeOutput) -> torch.Tensor:
    """
    Replicates the AvatarForcing dataset.py audio embedding build:

        audio_emb = hs.last_hidden_state
        for h in hs.hidden_states:
            audio_emb = torch.cat([audio_emb, h], dim=-1)

    Parameters
    ----------
    hs : Wav2Vec2LikeOutput  (output_hidden_states=True required)

    Returns
    -------
    audio_emb : (B, T, 10752)   i.e. 14 × 768
    """
    assert hs.hidden_states is not None, (
        "Call bridge with output_hidden_states=True to get hidden_states."
    )
    audio_emb = hs.last_hidden_state
    for h in hs.hidden_states:
        audio_emb = torch.cat([audio_emb, h], dim=-1)
    return audio_emb


def avatarforcing_prepend_zero(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Prepends a zero frame as done in AvatarForcing dataset.py:
        audio_emb = torch.cat([torch.zeros_like(audio_emb[:1]), audio_emb], dim=0)

    Parameters
    ----------
    audio_emb : (T, D) or (B, T, D)

    Returns
    -------
    (T+1, D) or (B, T+1, D)
    """
    if audio_emb.dim() == 2:
        return torch.cat([torch.zeros_like(audio_emb[:1]), audio_emb], dim=0)
    return torch.cat([torch.zeros_like(audio_emb[:, :1]), audio_emb], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Shared checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str, model: MimiHuBERTBridge, device: torch.device):
    """
    Load bridge weights from a checkpoint file.
    Supports both full trainer checkpoints ({"bridge": state_dict, ...})
    and bare state_dicts saved directly.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    sd = ckpt.get("bridge", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch Inference
# ──────────────────────────────────────────────────────────────────────────────

class BridgeInference:
    """
    Batch inference wrapper.
    Loads a trained bridge checkpoint and converts Mimi tokens → audio features.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device(self.cfg["inference"].get("device", "cuda"))
        else:
            self.device = torch.device("cpu")

        self.output_dim = self.cfg["model"]["output_dim"]      # 768
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()
        logger.info(
            f"Loaded bridge from {checkpoint_path} on {self.device} "
            f"(output_dim={self.output_dim}, 12 layers)"
        )

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,
        avatarforcing_format: bool = False,
        prepend_zero: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, T, num_codebooks) or (T, num_codebooks)  int64
        avatarforcing_format : if True, return 14-layer concat (B, 2T, 10752)
                               if False, return last_hidden_state (B, 2T, 768)
        prepend_zero : if True, prepend a zero frame (like AvatarForcing dataset.py)
                       Only applied when avatarforcing_format=True.

        Returns
        -------
        features : float32 on CPU
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.to(self.device)

        hs, _ = self.model(tokens, output_hidden_states=avatarforcing_format)

        if avatarforcing_format:
            features = avatarforcing_concat(hs)   # (B, 2T, 10752)
            if prepend_zero:
                # squeeze if single item, prepend, re-expand
                f = features.squeeze(0)           # (2T, 10752)
                f = avatarforcing_prepend_zero(f) # (2T+1, 10752)
                features = f.unsqueeze(0)         # (1, 2T+1, 10752)
        else:
            features = hs.last_hidden_state       # (B, 2T, 768)

        return features.float().cpu()

    @torch.no_grad()
    def from_audio(
        self,
        audio_path: str,
        avatarforcing_format: bool = False,
        prepend_zero: bool = False,
    ) -> torch.Tensor:
        """
        End-to-end: audio file → features.

        Returns
        -------
        (T_out, D) float32 tensor   D=768 or D=10752 depending on format flag
        """
        from dataset import MimiExtractor
        import torchaudio

        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        extractor = MimiExtractor(self.cfg["paths"]["mimi_model"])
        tokens = extractor.extract(waveform, native_sr)   # (T, 8)

        features = self(
            tokens,
            avatarforcing_format=avatarforcing_format,
            prepend_zero=prepend_zero,
        )                                                  # (1, T_out, D)
        return features.squeeze(0)                         # (T_out, D)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Inference
# ──────────────────────────────────────────────────────────────────────────────

class StreamingBridgeInference:
    """
    Causal streaming inference with KV-cache.

    Usage:
        stream = StreamingBridgeInference(checkpoint, config)
        stream.reset()
        for mimi_chunk in token_stream:           # (chunk_size, 8)
            feat_chunk = stream.step(mimi_chunk)  # (2*chunk_size, 768)
            # or with AvatarForcing format:
            feat_chunk = stream.step(mimi_chunk, avatarforcing_format=True)
            #                                       (2*chunk_size, 10752)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        chunk_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.chunk_size = chunk_size or self.cfg["inference"].get("chunk_size", 50)
        self.output_dim = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()

        self._past_kvs: Optional[list] = None
        self._step_count = 0

    def reset(self):
        """Reset streaming state (call at the start of each new utterance)."""
        self._past_kvs = None
        self._step_count = 0

    @torch.no_grad()
    def step(
        self,
        tokens_chunk: torch.Tensor,
        avatarforcing_format: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens_chunk : (C, 8) or (1, C, 8)  int64
        avatarforcing_format : if True, return (2C, 10752); else (2C, 768)

        Returns
        -------
        features : float32 on CPU
        """
        if tokens_chunk.dim() == 2:
            tokens_chunk = tokens_chunk.unsqueeze(0)   # (1, C, 8)

        tokens_chunk = tokens_chunk.to(self.device)

        hs, present_kvs = self.model(
            tokens_chunk,
            output_hidden_states=avatarforcing_format,
            use_cache=True,
            past_kvs=self._past_kvs,
        )
        self._past_kvs = present_kvs

        # Trim KV cache to prevent unbounded growth
        max_kv_len = self.cfg["model"].get("max_seq_len", 2048) * 2
        if self._past_kvs is not None:
            trimmed = []
            for layer_kv in self._past_kvs:
                if layer_kv is not None:
                    k, v = layer_kv
                    if k.shape[2] > max_kv_len:
                        k = k[:, :, -max_kv_len:]
                        v = v[:, :, -max_kv_len:]
                    trimmed.append((k, v))
                else:
                    trimmed.append(None)
            self._past_kvs = trimmed

        self._step_count += 1

        if avatarforcing_format:
            features = avatarforcing_concat(hs)  # (1, 2C, 10752)
        else:
            features = hs.last_hidden_state       # (1, 2C, 768)

        return features.squeeze(0).float().cpu()  # (2C, D)

    def stream_tokens(
        self,
        tokens: torch.Tensor,
        avatarforcing_format: bool = False,
    ) -> Iterator[torch.Tensor]:
        """
        Yield feature chunks from a full token sequence.
        tokens : (T, 8)
        yields : (2*chunk_size, D) tensors
        """
        self.reset()
        T = tokens.shape[0]
        for start in range(0, T, self.chunk_size):
            chunk = tokens[start : start + self.chunk_size]
            yield self.step(chunk, avatarforcing_format=avatarforcing_format)


# ──────────────────────────────────────────────────────────────────────────────
# Latency Benchmark Utility
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_streaming(
    checkpoint: str,
    config: str,
    num_chunks: int = 100,
    chunk_size: int = 50,
    warmup: int = 5,
):
    """Quick per-chunk latency benchmark for streaming mode."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    num_codebooks = cfg["model"]["num_codebooks"]
    output_dim    = cfg["model"]["output_dim"]

    stream = StreamingBridgeInference(checkpoint, config, chunk_size=chunk_size)
    dummy_tokens = torch.randint(0, cfg["model"]["vocab_size"], (chunk_size, num_codebooks))

    stream.reset()
    for _ in range(warmup):
        stream.step(dummy_tokens)

    stream.reset()
    times = []
    for _ in range(num_chunks):
        t0 = time.perf_counter()
        stream.step(dummy_tokens)
        if stream.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    audio_secs_per_chunk = chunk_size / cfg["data"]["mimi_rate"]
    print(f"\n=== Streaming Latency Benchmark ===")
    print(f"Chunk size   : {chunk_size} Mimi frames → {chunk_size * 2} feature frames")
    print(f"Audio / chunk: {audio_secs_per_chunk:.2f}s  |  output_dim={output_dim} | 12 layers")
    print(f"Median : {statistics.median(times):.2f} ms")
    print(f"P95    : {sorted(times)[int(0.95 * len(times))]:.2f} ms")
    print(f"Max    : {max(times):.2f} ms")
    print(
        f"Throughput: {1000 / statistics.median(times):.1f} chunks/s  "
        f"({1000 * audio_secs_per_chunk / statistics.median(times):.1f}× realtime)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mimi-to-Wav2Vec2 Bridge Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to config.yaml")
    parser.add_argument("--audio",      default=None,  help="Input audio file")
    parser.add_argument("--tokens",     default=None,  help="Pre-extracted .pt token file (T, 8)")
    parser.add_argument("--output",     default="features.pt", help="Output .pt file path")
    parser.add_argument("--streaming",  action="store_true", help="Use causal streaming mode")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size in Mimi frames")
    parser.add_argument("--benchmark",  action="store_true", help="Run latency benchmark")
    parser.add_argument("--device",     default=None,  help="Force device (cuda / cpu)")
    parser.add_argument(
        "--avatarforcing",
        action="store_true",
        help=(
            "Output 14-layer concatenated format (B, T, 10752) ready for "
            "AvatarForcing diffusion model. Adds a zero prefix frame."
        ),
    )

    # ── Compare mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare real wav2vec2 output vs bridge model prediction for --audio."
    )
    parser.add_argument("--hubert-model", default=None)
    parser.add_argument("--mimi-model",   default=None)
    parser.add_argument("--save-gt",      default=None)
    parser.add_argument("--save-pred",    default=None)
    parser.add_argument("--save-gt-npy",   default="wav2vec_gt_features.npy")
    parser.add_argument("--save-pred-npy", default="bridge_pred_features.npy")
    parser.add_argument("--no-auto-save-npy", action="store_true")
    parser.add_argument("--plot",         action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.benchmark:
        benchmark_streaming(args.checkpoint, args.config, chunk_size=args.chunk_size)
        return

    if args.compare:
        if args.audio is None:
            parser.error("--compare requires --audio")
        from compare_inference import compare as run_compare
        run_compare(
            audio_path            = args.audio,
            checkpoint_path       = args.checkpoint,
            config_path           = args.config,
            device                = args.device,
            hubert_model_override = args.hubert_model,
            mimi_model_override   = args.mimi_model,
            save_gt               = args.save_gt,
            save_pred             = args.save_pred,
            save_gt_npy           = args.save_gt_npy,
            save_pred_npy         = args.save_pred_npy,
            auto_save_npy         = not args.no_auto_save_npy,
            plot                  = args.plot,
        )
        return

    if args.audio is None and args.tokens is None:
        parser.error("Provide at least one of --audio or --tokens")

    def get_tokens(cfg: dict) -> torch.Tensor:
        if args.tokens:
            try:
                return torch.load(args.tokens, weights_only=True)
            except TypeError:
                return torch.load(args.tokens)
        from dataset import MimiExtractor
        import torchaudio
        wav, native_sr = torchaudio.load(args.audio)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return MimiExtractor(cfg["paths"]["mimi_model"]).extract(wav, native_sr)

    if args.streaming:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        tokens = get_tokens(cfg)

        stream = StreamingBridgeInference(
            args.checkpoint, args.config,
            chunk_size=args.chunk_size, device=args.device,
        )
        chunks = list(
            stream.stream_tokens(tokens, avatarforcing_format=args.avatarforcing)
        )
        features = torch.cat(chunks, dim=0)
        logger.info(f"Streaming output: {features.shape}")

    else:
        infer = BridgeInference(args.checkpoint, args.config, device=args.device)
        if args.audio:
            features = infer.from_audio(
                args.audio,
                avatarforcing_format=args.avatarforcing,
                prepend_zero=args.avatarforcing,
            )
        else:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            tokens = get_tokens(cfg)
            features = infer(
                tokens,
                avatarforcing_format=args.avatarforcing,
                prepend_zero=args.avatarforcing,
            ).squeeze(0)
        logger.info(f"Batch output: {features.shape}")

    torch.save(features, args.output)
    logger.info(f"Saved features → {args.output}")


if __name__ == "__main__":
    main()

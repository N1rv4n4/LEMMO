from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import P16SignalEncoder, RMSNorm, sinusoidal_positions


class QuestionConditionedQueryBridge(nn.Module):
    def __init__(self, query_count: int = 16) -> None:
        super().__init__()
        signal_dim, qwen_dim, heads = 768, 2560, 8
        self.base_queries = nn.Parameter(torch.empty(query_count, signal_dim))
        self.question_input_norm = nn.LayerNorm(qwen_dim)
        self.question_down = nn.Linear(qwen_dim, signal_dim)
        self.question_query_norm = nn.LayerNorm(signal_dim)
        self.question_key_norm = nn.LayerNorm(signal_dim)
        self.question_attention = nn.MultiheadAttention(signal_dim, heads, batch_first=True)
        self.question_ffn_norm = nn.LayerNorm(signal_dim)
        self.question_ffn = nn.Sequential(
            nn.Linear(signal_dim, signal_dim * 4), nn.GELU(), nn.Linear(signal_dim * 4, signal_dim)
        )
        self.signal_query_norm = nn.LayerNorm(signal_dim)
        self.signal_key_norm = nn.LayerNorm(signal_dim)
        self.signal_attention = nn.MultiheadAttention(signal_dim, heads, batch_first=True)
        self.signal_ffn_norm = nn.LayerNorm(signal_dim)
        self.signal_ffn = nn.Sequential(
            nn.Linear(signal_dim, signal_dim * 4), nn.GELU(), nn.Linear(signal_dim * 4, signal_dim)
        )
        self.output_norm = nn.LayerNorm(signal_dim)
        self.projector = nn.Linear(signal_dim, qwen_dim)

    def forward(
        self,
        signal_tokens: torch.Tensor,
        signal_mask: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = signal_tokens.shape[0]
        queries = self.base_queries[None].expand(batch, -1, -1)
        question = self.question_down(self.question_input_norm(question_embeddings))
        conditioned, _ = self.question_attention(
            self.question_query_norm(queries),
            self.question_key_norm(question),
            self.question_key_norm(question),
            key_padding_mask=~question_mask.bool(),
            need_weights=False,
        )
        queries = queries + conditioned
        queries = queries + self.question_ffn(self.question_ffn_norm(queries))
        signal = self.signal_key_norm(signal_tokens)
        extracted, _ = self.signal_attention(
            self.signal_query_norm(queries),
            signal,
            signal,
            key_padding_mask=~signal_mask.bool(),
            need_weights=False,
        )
        queries = queries + extracted
        queries = queries + self.signal_ffn(self.signal_ffn_norm(queries))
        return {
            "conditioned_queries": queries,
            "qwen_query_embeddings": self.projector(self.output_norm(queries)),
        }


class NativeBidirectionalLinearAttention(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, head_dim: int = 64) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        q = F.elu(self.q_proj(x).view(shape).float()) + 1.0
        k = F.elu(self.k_proj(x).view(shape).float()) + 1.0
        v = self.v_proj(x).view(shape).float()
        kv = torch.einsum("bnhd,bnhe->bhde", k, v)
        numerator = torch.einsum("bnhd,bhde->bnhe", q, kv)
        denominator = torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)).clamp_min(1e-6)
        return self.out_proj((numerator / denominator[..., None]).reshape(batch, tokens, dim).to(x.dtype))


class ResamplerSwiGLU(nn.Module):
    """SwiGLU with names matching the released resampler state dictionary."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class FusionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm(768)
        self.attention = NativeBidirectionalLinearAttention()
        self.norm2 = RMSNorm(768)
        self.ffn = ResamplerSwiGLU(768, 1536)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ConnectedChunkDownsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim, stages, groups = 768, 6, 96
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("anti_alias_kernel", kernel.view(1, 1, 5).repeat(dim, 1, 1), persistent=True)
        self.corrections = nn.ModuleList(
            [nn.Conv1d(dim, dim, 5, stride=2, padding=2, groups=groups, bias=False) for _ in range(stages)]
        )
        self.correction_scales = nn.Parameter(torch.full((stages,), 0.01))
        self.output_norm = RMSNorm(dim)
        self.output_ffn = ResamplerSwiGLU(dim, dim // 2)

    def forward(
        self, hidden: torch.Tensor, output_tokens: int | None = None
    ) -> torch.Tensor:
        requested_tokens = 800 if output_tokens is None else int(output_tokens)
        if requested_tokens < 1:
            raise ValueError("output_tokens must be positive")
        x = hidden.transpose(1, 2)
        for index, correction in enumerate(self.corrections):
            base = F.conv1d(x.float(), self.anti_alias_kernel.float(), stride=2, padding=2, groups=768)
            residual = correction(x.to(correction.weight.dtype)).float()
            x = (base + torch.tanh(self.correction_scales[index].float()) * residual).to(hidden.dtype)
        x = F.adaptive_avg_pool1d(x.float(), requested_tokens).transpose(1, 2).to(hidden.dtype)
        return x + self.output_ffn(self.output_norm(x))


class LongSignalResampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_downsampler = ConnectedChunkDownsampler()
        self.chunk_embeddings = nn.Parameter(torch.empty(5, 768))
        # State compatibility with the released resampler. Not used in inference.
        self.mask_token = nn.Parameter(torch.empty(768))
        self.fusion_blocks = nn.ModuleList([FusionBlock(), FusionBlock()])
        self.fusion_norm = RMSNorm(768)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        batch, count, tokens, dim = chunks.shape
        compressed = self.chunk_downsampler(chunks.reshape(batch * count, tokens, dim))
        flat = compressed.reshape(batch, count * 800, dim)
        chunk_ids = torch.arange(count, device=flat.device).repeat_interleave(800)
        flat = flat + self.chunk_embeddings[chunk_ids][None].to(flat)
        flat = flat + sinusoidal_positions(
            torch.arange(count * 800, device=flat.device), dim, flat.dtype
        )[None]
        for block in self.fusion_blocks:
            flat = block(flat)
        return self.fusion_norm(flat)


class VariableLongSignalResampler(nn.Module):
    """Variable-tail resampler for one to eight continuous 1M-point chunks."""

    def __init__(self, maximum_chunks: int = 8) -> None:
        super().__init__()
        self.maximum_chunks = int(maximum_chunks)
        self.chunk_downsampler = ConnectedChunkDownsampler()
        self.chunk_embeddings = nn.Parameter(torch.empty(self.maximum_chunks, 768))
        self.mask_token = nn.Parameter(torch.empty(768))
        self.fusion_blocks = nn.ModuleList([FusionBlock(), FusionBlock()])
        self.fusion_norm = RMSNorm(768)

    @staticmethod
    def _output_tokens(input_tokens: int) -> int:
        if not 1 <= int(input_tokens) <= 62_500:
            raise ValueError("chunk token length must be in [1,62500]")
        return max(1, round(800 * int(input_tokens) / 62_500))

    def forward(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        if not 1 <= len(chunks) <= self.maximum_chunks:
            raise ValueError("chunks must contain between one and eight tensors")
        compressed: list[torch.Tensor] = []
        lengths: list[int] = []
        for chunk in chunks:
            if chunk.ndim != 2 or chunk.shape[-1] != 768:
                raise ValueError("each chunk must have shape [N,768]")
            output_tokens = self._output_tokens(int(chunk.shape[0]))
            compressed.append(
                self.chunk_downsampler(chunk[None], output_tokens=output_tokens)[0]
            )
            lengths.append(output_tokens)
        flat = torch.cat(compressed, dim=0)[None]
        chunk_ids = torch.repeat_interleave(
            torch.arange(len(chunks), device=flat.device),
            torch.tensor(lengths, device=flat.device),
        )
        flat = flat + self.chunk_embeddings[chunk_ids][None].to(flat)
        flat = flat + sinusoidal_positions(
            torch.arange(flat.shape[1], device=flat.device), 768, flat.dtype
        )[None]
        for block in self.fusion_blocks:
            flat = block(flat)
        return self.fusion_norm(flat)


class EncoderWrapper(nn.Module):
    """Preserves the checkpoint key prefix ``encoder.encoder.*``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = P16SignalEncoder()

    def forward_tokens(
        self, patches: torch.Tensor, lengths: torch.Tensor, sample_rates: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder.forward_patches(patches, lengths, sample_rates)


class EMLMRecognitionSignalPath(nn.Module):
    def __init__(self, query_count: int = 16) -> None:
        super().__init__()
        self.encoder = EncoderWrapper()
        self.resampler = LongSignalResampler()
        self.bridge = QuestionConditionedQueryBridge(query_count)

    def encode(self, signals: torch.Tensor, sample_rates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = signals.shape
        if channels != 2:
            raise ValueError("signals must have shape [B,L,2]")
        if length == 5_000_000:
            chunks = signals.reshape(batch * 5, 1_000_000, 2)
            patches = chunks.reshape(batch * 5, 62_500, 32).reshape(-1, 32)
            lengths = torch.full((batch * 5,), 62_500, dtype=torch.long, device=signals.device)
            hidden = self.encoder.forward_tokens(patches, lengths, sample_rates.repeat_interleave(5))
            tokens = self.resampler(hidden.reshape(batch, 5, 62_500, 768))
        else:
            if length % 16:
                raise ValueError("IQ length must be divisible by 16")
            count = length // 16
            patches = signals.reshape(batch, count, 32).reshape(-1, 32)
            lengths = torch.full((batch,), count, dtype=torch.long, device=signals.device)
            tokens = self.encoder.forward_tokens(patches, lengths, sample_rates).reshape(batch, count, 768)
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)


class ResidualTaskAdapter(nn.Module):
    def __init__(self, bottleneck: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.down = nn.Linear(768, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, 768, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class EMLMRecognitionModel(nn.Module):
    TASKS = ("amc", "interference", "uav", "wtc")

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.signal = EMLMRecognitionSignalPath(int(config.get("query_count", 16)))
        bottleneck = int(config.get("adapter_bottleneck", 64))
        self.task_adapters = nn.ModuleDict({task: ResidualTaskAdapter(bottleneck) for task in self.TASKS})

    def forward(
        self,
        signals: torch.Tensor,
        sample_rates: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        tokens, mask = self.signal.encode(signals, sample_rates)
        bridge = self.signal.bridge(tokens, mask, question_embeddings, question_mask)
        adapted = self.task_adapters[task](bridge["conditioned_queries"])
        return self.signal.bridge.projector(self.signal.bridge.output_norm(adapted))


class EMLMDescriptionModel(nn.Module):
    patch_size = 16
    chunk_points = 1_000_000
    maximum_chunks = 8

    def __init__(self, query_count: int = 64) -> None:
        super().__init__()
        self.encoder = EncoderWrapper()
        self.resampler = VariableLongSignalResampler(self.maximum_chunks)
        self.bridge = QuestionConditionedQueryBridge(int(query_count))

    def encode(self, signals: torch.Tensor, sample_rates: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3 or signals.shape[0] != 1 or signals.shape[-1] != 2:
            raise ValueError("EMLM_Description currently expects one [1,L,2] signal")
        points = int(signals.shape[1])
        if points < self.patch_size or points > self.chunk_points * self.maximum_chunks:
            raise ValueError("EMLM_Description IQ length must be in [16,8000000]")
        chunks: list[torch.Tensor] = []
        for start in range(0, points, self.chunk_points):
            current = signals[:, start : min(points, start + self.chunk_points)]
            usable = int(current.shape[1]) - int(current.shape[1]) % self.patch_size
            if usable < self.patch_size:
                continue
            token_count = usable // self.patch_size
            patches = current[:, :usable].reshape(token_count, self.patch_size * 2)
            lengths = torch.tensor([token_count], dtype=torch.long, device=signals.device)
            chunks.append(self.encoder.forward_tokens(patches, lengths, sample_rates))
        if not chunks:
            raise RuntimeError("Signal has no complete P16 patch")
        return chunks[0][None] if points <= self.chunk_points else self.resampler(chunks)

    def forward(
        self,
        signals: torch.Tensor,
        sample_rates: torch.Tensor,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.encode(signals, sample_rates)
        signal_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return self.bridge(
            tokens, signal_mask, question_embeddings, question_mask
        )["qwen_query_embeddings"]


def load_emlm_recognition_weights(model: EMLMRecognitionModel, checkpoint: Path) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state" in saved:
        model.load_state_dict(saved["model_state"], strict=True)
        return saved
    signal_state = saved.get("signal_state", saved.get("signal_trainable_state"))
    if signal_state is None:
        raise KeyError("EMLM_Recognition inference checkpoint has no signal state")
    incompatible = model.signal.load_state_dict(signal_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected EMLM_Recognition signal keys: {incompatible.unexpected_keys}")
    branch_state = saved["task_branch_state"]
    for task, adapter in model.task_adapters.items():
        prefix = f"{task}.adapter."
        state = {key[len(prefix):]: value for key, value in branch_state.items() if key.startswith(prefix)}
        adapter.load_state_dict(state, strict=True)
    return saved


def load_emlm_description_weights(model: EMLMDescriptionModel, checkpoint: Path) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "emlm_description_inference_v2":
        raise RuntimeError("Unsupported EMLM_Description inference checkpoint format")
    model.load_state_dict(saved.get("model_state", saved.get("signal_state")), strict=True)
    return saved


from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return x * scale.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LinearMixer(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        from fla.layers import LinearAttention

        self.linear_attention = LinearAttention(
            mode="chunk",
            hidden_size=hidden_size,
            expand_k=0.5,
            expand_v=1.0,
            num_heads=num_heads,
            feature_map="elu",
            output_norm="rmsnorm",
            do_feature_map_norm=True,
            norm_eps=1e-5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_attention(
            hidden_states=x,
            attention_mask=None,
            use_cache=False,
            output_attentions=False,
        )[0]


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, ffn_size: int = 3072) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.mixer = LinearMixer(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class SamplingRateEmbedder(nn.Module):
    def __init__(self, hidden_size: int = 768, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = int(frequency_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, sample_rates: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=sample_rates.device, dtype=torch.float32)
            / float(half)
        )
        angles = sample_rates.float()[:, None] * frequencies[None]
        features = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.mlp(features.to(self.mlp[0].weight.dtype))


def sinusoidal_positions(
    positions: torch.Tensor, hidden_size: int = 768, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    half = hidden_size // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=positions.device, dtype=torch.float32)
        / float(max(1, half))
    )
    angles = positions.float()[:, None] * frequencies[None]
    encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if encoded.shape[-1] < hidden_size:
        encoded = F.pad(encoded, (0, hidden_size - encoded.shape[-1]))
    return encoded.to(dtype=dtype)


class P16SignalEncoder(nn.Module):
    """Forward-only P16 electromagnetic encoder used by both released runtimes."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 16
        self.patch_dim = 32
        self.hidden_dim = 768
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.empty(self.hidden_dim))
        self.fs_embed = SamplingRateEmbedder(self.hidden_dim)
        self.encoder_embed_norm = RMSNorm(self.hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(self.hidden_dim, 12, 3072) for _ in range(12)]
        )
        self.encoder_norm = RMSNorm(self.hidden_dim)
        # Kept for state compatibility; it is never used during inference.
        self.encoder_mask_token = nn.Parameter(torch.empty(self.hidden_dim))

    @staticmethod
    def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(1, device=lengths.device, dtype=torch.int32),
                torch.cumsum(lengths.to(torch.int32), dim=0),
            )
        )

    @staticmethod
    def _layout(lengths: torch.Tensor):
        device = lengths.device
        samples = lengths.numel()
        iq_total = int(lengths.sum().item())
        full_lengths = lengths + 2
        full_offsets = torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(full_lengths.long(), dim=0)[:-1],
            )
        )
        patch_sample_ids = torch.repeat_interleave(torch.arange(samples, device=device), lengths.long())
        iq_offsets = torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(lengths.long(), dim=0)[:-1],
            )
        )
        local_iq = torch.arange(iq_total, device=device) - iq_offsets[patch_sample_ids]
        iq_indices = full_offsets[patch_sample_ids] + 2 + local_iq
        full_sample_ids = torch.repeat_interleave(torch.arange(samples, device=device), full_lengths.long())
        positions = torch.arange(int(full_lengths.sum().item()), device=device) - full_offsets[full_sample_ids]
        return full_lengths, full_offsets, iq_indices, positions

    def _run_blocks(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        output_indices: list[torch.Tensor] = []
        output_values: list[torch.Tensor] = []
        for length_tensor in torch.unique(lengths, sorted=True):
            length = int(length_tensor.item())
            sample_ids = torch.nonzero(lengths.eq(length_tensor), as_tuple=False).flatten()
            for first in range(0, sample_ids.numel(), 1024):
                chunk_ids = sample_ids[first : first + 1024]
                starts = cu_seqlens[chunk_ids].long()
                flat = starts[:, None] + torch.arange(length, device=x.device)[None]
                hidden = x[flat]
                real_batch = hidden.shape[0]
                bucket = 1 if real_batch == 1 else 1 << (real_batch - 1).bit_length()
                if bucket > real_batch:
                    hidden = torch.cat(
                        (hidden, hidden.new_zeros((bucket - real_batch, length, hidden.shape[-1])))
                    )
                for block in self.encoder_blocks:
                    hidden = block(hidden)
                output_indices.append(flat.reshape(-1))
                output_values.append(hidden[:real_batch].reshape(-1, self.hidden_dim))
        return torch.zeros_like(x).index_copy(0, torch.cat(output_indices), torch.cat(output_values))

    def forward_patches(
        self, patches: torch.Tensor, lengths: torch.Tensor, sample_rates: torch.Tensor
    ) -> torch.Tensor:
        lengths = lengths.long()
        full_lengths, full_offsets, iq_indices, positions = self._layout(lengths)
        full = patches.new_empty((int(full_lengths.sum()), self.hidden_dim))
        full[iq_indices] = self.patch_embed(patches)
        full[full_offsets] = self.cls_token.to(full)
        full[full_offsets + 1] = self.fs_embed(sample_rates).to(full)
        full = self.encoder_embed_norm(
            full + sinusoidal_positions(positions, self.hidden_dim, full.dtype)
        )
        hidden = self._run_blocks(full, self._cu_seqlens(full_lengths))
        return self.encoder_norm(hidden)[iq_indices]

    def load_base_checkpoint(self, checkpoint: str) -> None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved.get("model", saved)
        own = self.state_dict()
        copied = {}
        for key, value in own.items():
            if key == "encoder_mask_token" and key not in state:
                copied[key] = value
            elif key in state:
                copied[key] = state[key]
            else:
                raise KeyError(f"Encoder checkpoint is missing {key}")
        self.load_state_dict(copied, strict=True)


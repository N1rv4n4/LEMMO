from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(base.in_features, int(rank), bias=False)
        self.lora_B = nn.Linear(int(rank), base.out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = self.lora_B(self.lora_A(self.dropout(x).to(self.lora_A.weight.dtype)))
        return base + update.to(base.dtype) * self.scaling


DEFAULT_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
    "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj",
}


def inject_qwen_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0.0) -> None:
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not (
            module_name.startswith("model.layers.")
            or ".model.layers." in module_name
            or "language_model.layers." in module_name
        ):
            continue
        for child_name, child in list(module.named_children()):
            if child_name in DEFAULT_TARGETS and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(f"{module_name}.{child_name}")
    if not replaced:
        raise RuntimeError("No supported Qwen linear layers were found for LoRA injection")


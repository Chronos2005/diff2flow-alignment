"""Minimal LoRA (Low-Rank Adaptation) for Linear and Conv2d layers."""

import math

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.lora_down = nn.Linear(original.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x))


class LoRAConv2d(nn.Module):
    def __init__(self, original: nn.Conv2d, rank: int):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.lora_down = nn.Conv2d(original.in_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, original.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x))


def apply_lora(model, rank=64, placement="all"):
    """
    Apply LoRA to Linear and Conv2d layers in the model.

    placement: "all"         — wrap every Linear and Conv2d layer
               "attention"   — wrap only layers inside attention blocks
                               (module path contains "attentions")
               "feedforward" — wrap all other layers (conv, time embedding, etc.)

    Freezes all original weights first, then adds trainable low-rank deltas
    only to the selected layers.
    Returns the number of replaced layers.
    """
    # Freeze everything up front; LoRA modules will unfreeze their delta params.
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1] if parts else None

        if not attr_name:
            continue

        # Filter by placement using the module path.
        # In HF UNet2DModel, attention layers have "attentions" as a path component
        # (e.g. down_blocks.0.attentions.0.to_q). Everything else is feedforward.
        has_attn = "attentions" in parts
        if placement == "attention" and not has_attn:
            continue
        if placement == "feedforward" and has_attn:
            continue

        if isinstance(module, nn.Linear):
            r = min(rank, module.in_features, module.out_features)
            setattr(parent, attr_name, LoRALinear(module, r))
            replaced += 1
        elif isinstance(module, nn.Conv2d):
            r = min(rank, module.in_channels, module.out_channels)
            setattr(parent, attr_name, LoRAConv2d(module, r))
            replaced += 1

    return replaced

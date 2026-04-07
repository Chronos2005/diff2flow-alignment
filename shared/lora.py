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


def apply_lora(model, rank=64):
    """
    Apply LoRA to all Linear and Conv2d layers in the model.
    Freezes original weights and adds trainable low-rank deltas.
    Returns the number of replaced layers.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1] if parts else None

        if attr_name and isinstance(module, nn.Linear):
            r = min(rank, module.in_features, module.out_features)
            setattr(parent, attr_name, LoRALinear(module, r))
            replaced += 1
        elif attr_name and isinstance(module, nn.Conv2d):
            r = min(rank, module.in_channels, module.out_channels)
            setattr(parent, attr_name, LoRAConv2d(module, r))
            replaced += 1

    return replaced

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

        self.lora_down = nn.Conv2d(
            original.in_channels, rank, 1,
            stride=original.stride, bias=False,
        )
        self.lora_up = nn.Conv2d(rank, original.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.original(x) + self.lora_up(self.lora_down(x))


def merge_lora(model):
    """
    Merge LoRA deltas into their base weights and restore plain layers.

    After merging, LoRALinear/LoRAConv2d wrappers are replaced with ordinary
    nn.Linear/nn.Conv2d whose weights absorb the trained low-rank update.
    The resulting model is identical in behaviour but has no LoRA overhead and
    can be saved/loaded with the standard UNet2DModel from_pretrained API.
    """
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if not parts:
            continue
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]

        if isinstance(module, LoRALinear):
            merged_weight = (
                module.original.weight.data
                + module.lora_up.weight.data @ module.lora_down.weight.data
            )
            new_layer = nn.Linear(
                module.original.in_features,
                module.original.out_features,
                bias=module.original.bias is not None,
            )
            new_layer.weight.data.copy_(merged_weight)
            if module.original.bias is not None:
                new_layer.bias.data.copy_(module.original.bias.data)
            setattr(parent, attr_name, new_layer)

        elif isinstance(module, LoRAConv2d):
            # Both lora_down and lora_up are 1×1 convs; merge as matrix multiply
            w_down = module.lora_down.weight.data.squeeze(-1).squeeze(-1)  # (rank, C_in)
            w_up = module.lora_up.weight.data.squeeze(-1).squeeze(-1)     # (C_out, rank)
            delta = (w_up @ w_down).unsqueeze(-1).unsqueeze(-1)           # (C_out, C_in, 1, 1)
            orig = module.original
            new_layer = nn.Conv2d(
                orig.in_channels, orig.out_channels, orig.kernel_size,
                stride=orig.stride, padding=orig.padding,
                dilation=orig.dilation, groups=orig.groups,
                bias=orig.bias is not None,
            )
            new_layer.weight.data.copy_(orig.weight.data + delta)
            if orig.bias is not None:
                new_layer.bias.data.copy_(orig.bias.data)
            setattr(parent, attr_name, new_layer)


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

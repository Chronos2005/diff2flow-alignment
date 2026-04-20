"""LoRA (Low-Rank Adaptation) using the standard PEFT library.

Wraps a diffusers UNet2DModel (or any nn.Module containing nn.Linear / nn.Conv2d
layers) with PEFT LoRA adapters. The wrapped model behaves like the original at
inference (forward signature preserved by PeftModel) but only the low-rank
adapter parameters are trainable.

Compared to a hand-rolled implementation this gives us, for free:
  - standard `lora_alpha / r` scaling (so rank changes don't bake in extra gain)
  - kernel-matching Conv2d LoRA (`lora_A` shares kernel size with the base conv,
    `lora_B` is 1x1) — strictly more expressive than 1x1-only point-wise LoRA
  - a well-tested merge path (`merge_and_unload`) so saved checkpoints exactly
    match the trained behaviour
"""

import copy

import torch.nn as nn
from peft import LoraConfig, get_peft_model


def _collect_target_modules(model, placement):
    """Walk the (un-wrapped) model and return module names matching the placement filter.

    placement: "all"         — every nn.Linear and nn.Conv2d
               "attention"   — only modules whose path contains "attentions"
                               (HF UNet2DModel convention)
               "feedforward" — every other nn.Linear / nn.Conv2d
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        has_attn = "attentions" in name.split(".")
        if placement == "attention" and not has_attn:
            continue
        if placement == "feedforward" and has_attn:
            continue
        targets.append(name)
    return targets


def apply_lora(model, rank=64, placement="all", alpha=None):
    """
    Wrap `model` with PEFT LoRA adapters on the selected Linear/Conv2d layers.

    Returns (peft_model, num_replaced). The returned model has the original
    forward signature preserved — call it just like the un-wrapped UNet.

    `alpha` defaults to `rank`, which gives a scale of 1.0 (matching the
    behaviour of a "no scaling" custom LoRA implementation). Setting
    `alpha == rank` is a common convention; varying alpha lets you decouple
    update magnitude from rank.
    """
    if alpha is None:
        alpha = rank

    targets = _collect_target_modules(model, placement)
    if not targets:
        return model, 0

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,  # kaiming-uniform on A, zeros on B
    )
    peft_model = get_peft_model(model, config)
    return peft_model, len(targets)


def save_merged(peft_model, save_dir):
    """
    Save the underlying base model with LoRA deltas merged in, without mutating
    the live training model.

    `merge_and_unload` rewrites the base layers' weights in place and removes
    the adapter, so calling it on the live model would invalidate the
    optimizer's parameter references and silently freeze further training.
    Deep-copying first keeps the live model and its optimizer state intact.
    """
    model_copy = copy.deepcopy(peft_model)
    merged = model_copy.merge_and_unload()
    merged.save_pretrained(save_dir)

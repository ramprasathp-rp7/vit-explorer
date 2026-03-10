# backend/rollout.py
# Attention Rollout + LRP logic, adapted from the project's training code.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Core rollout engine
# ─────────────────────────────────────────────────────────────────────────────

class ViTAttentionRolloutLRP:
    """
    Computes the 'rollout' of attention relevance from the CLS token to all patches.
    Based on Chefer et al. / standard Transformer LRP.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.0):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    @torch.no_grad()
    def generate_lrp(self, inputs: torch.Tensor, target_class=None):
        outputs = self.model(
            pixel_values=inputs,
            output_attentions=True,
            output_hidden_states=False,
        )
        logits = outputs.logits
        attn_maps = outputs.attentions  # Tuple of [B, Heads, Seq, Seq]

        # Stack layers → [Layers, B, Heads, Seq, Seq], take batch 0
        attn_tensor = torch.stack(attn_maps)[:, 0]  # [Layers, Heads, Seq, Seq]

        # Fuse heads
        if self.head_fusion == "mean":
            attn_tensor = attn_tensor.mean(dim=1)   # [Layers, Seq, Seq]
        else:
            attn_tensor = attn_tensor.max(dim=1)[0]

        # Rollout: start from identity, multiply through layers
        result = torch.eye(attn_tensor.size(-1), device=attn_tensor.device)

        for i in range(attn_tensor.size(0)):
            attn_layer = attn_tensor[i].clone()

            # Discard low-attention noise
            if self.discard_ratio > 0:
                flat = attn_layer.view(-1)
                val, _ = torch.topk(flat, int(flat.size(0) * (1 - self.discard_ratio)))
                attn_layer[attn_layer < val[-1]] = 0

            # Add residual + re-normalize
            attn_layer = attn_layer + torch.eye(attn_layer.size(-1), device=attn_layer.device)
            attn_layer = attn_layer / attn_layer.sum(dim=-1, keepdim=True)

            result = torch.matmul(attn_layer, result)

        # CLS → patch relevance (index 0 → indices 1:)
        mask = result[0, 1:]
        return mask, logits


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def _smooth_map_tensor(tensor2d: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    if kernel_size <= 1:
        return tensor2d

    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gauss1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss1d = gauss1d / gauss1d.sum()

    k_row = gauss1d.view(1, 1, 1, kernel_size).to(tensor2d.device)
    k_col = gauss1d.view(1, 1, kernel_size, 1).to(tensor2d.device)

    x = tensor2d.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, k_row)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, k_col)
    return x.squeeze()


def generate_patch_map(
    model,
    pixel_values: torch.Tensor,
    discard_ratio: float = 0.05,
    head_fusion: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (patch_map [H, W] float 0-1, logits [1, C])."""
    lrp = ViTAttentionRolloutLRP(model, head_fusion=head_fusion, discard_ratio=discard_ratio)
    relevance_vec, logits = lrp.generate_lrp(inputs=pixel_values)

    num_patches_side = int(pixel_values.shape[-1] // 16)  # 224 / 16 = 14
    patch_map = relevance_vec.reshape(num_patches_side, num_patches_side)
    patch_map = _normalize_map(patch_map)
    patch_map = _smooth_map_tensor(patch_map, kernel_size=3, sigma=0.8)
    return patch_map, logits


# ─────────────────────────────────────────────────────────────────────────────
# Overlay builder (returns PIL Image)
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay(
    image_pil: Image.Image,
    patch_map: torch.Tensor,
    alpha: float = 0.55,
) -> Image.Image:
    """
    Overlay a heatmap (14×14 tensor, values 0-1) on a PIL image.
    Returns a PIL Image of the same size as image_pil.
    """
    img_np = np.array(image_pil.convert("RGB")).astype(np.uint8)
    h, w = img_np.shape[:2]

    heatmap_np = patch_map.cpu().numpy()
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_CUBIC)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)


def build_raw_heatmap(
    image_pil: Image.Image,
    patch_map: torch.Tensor,
) -> Image.Image:
    """Returns just the upscaled colormap as a PIL Image (no blending)."""
    h, w = image_pil.size[1], image_pil.size[0]
    heatmap_np = patch_map.cpu().numpy()
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(heatmap_colored)

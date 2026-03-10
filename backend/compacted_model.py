"""
Compacted ViT architecture and loader.
Ported from local_improved_physical_compaction_robust_v1_final.py.
Only the architecture definition and load_compacted_model() are needed here —
no weight transfer or training code.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class LayerConfig:
    num_heads: int
    mlp_intermediate_dim: int

    def __post_init__(self):
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.mlp_intermediate_dim < 1:
            raise ValueError(f"mlp_intermediate_dim must be >= 1, got {self.mlp_intermediate_dim}")


@dataclass
class CompactedViTConfig:
    hidden_size: int = 768
    num_layers: int = 12
    patch_size: int = 16
    image_size: int = 224
    num_channels: int = 3
    num_labels: int = 10
    layer_configs: List[LayerConfig] = None
    num_patches: int = None

    def __post_init__(self):
        if self.layer_configs is None:
            raise ValueError("layer_configs must be provided")
        if len(self.layer_configs) != self.num_layers:
            raise ValueError(
                f"layer_configs length ({len(self.layer_configs)}) must match "
                f"num_layers ({self.num_layers})"
            )
        self.num_patches = (self.image_size // self.patch_size) ** 2

    def to_dict(self) -> Dict:
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'patch_size': self.patch_size,
            'image_size': self.image_size,
            'num_channels': self.num_channels,
            'num_labels': self.num_labels,
            'layer_configs': [
                {'num_heads': lc.num_heads, 'mlp_intermediate_dim': lc.mlp_intermediate_dim}
                for lc in self.layer_configs
            ]
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CompactedViTConfig':
        layer_configs = [
            LayerConfig(
                num_heads=lc['num_heads'],
                mlp_intermediate_dim=lc['mlp_intermediate_dim']
            )
            for lc in config_dict['layer_configs']
        ]
        return cls(
            hidden_size=config_dict['hidden_size'],
            num_layers=config_dict['num_layers'],
            patch_size=config_dict['patch_size'],
            image_size=config_dict['image_size'],
            num_channels=config_dict['num_channels'],
            num_labels=config_dict['num_labels'],
            layer_configs=layer_configs
        )


# ==============================================================================
# ARCHITECTURE
# ==============================================================================

class CompactedViTEmbeddings(nn.Module):
    def __init__(self, config: CompactedViTConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(0.0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class CompactedViTSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        original_num_heads = 12  # ViT-Base fixed constant
        self.head_dim = hidden_size // original_num_heads  # Always 64
        self.num_heads = num_heads
        self.all_head_size = num_heads * self.head_dim
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.0)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_shape)


class CompactedViTSelfOutput(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        original_num_heads = 12
        head_dim = hidden_size // original_num_heads  # 64
        all_head_size = num_heads * head_dim
        self.dense = nn.Linear(all_head_size, hidden_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.dropout(hidden_states)


class CompactedViTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = CompactedViTSelfAttention(hidden_size, num_heads)
        self.output = CompactedViTSelfOutput(hidden_size, num_heads)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layernorm_before(hidden_states)
        attention_output = self.attention(normed)
        attention_output = self.output(attention_output, hidden_states)
        return hidden_states + attention_output


class CompactedViTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.dropout(hidden_states)


class CompactedViTLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_intermediate_dim: int):
        super().__init__()
        self.attention = CompactedViTAttention(hidden_size, num_heads)
        self.layernorm_before_mlp = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = CompactedViTMLP(hidden_size, mlp_intermediate_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attention(hidden_states)
        residual = hidden_states
        hidden_states = self.layernorm_before_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CompactedViTEncoder(nn.Module):
    def __init__(self, config: CompactedViTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CompactedViTLayer(
                hidden_size=config.hidden_size,
                num_heads=layer_config.num_heads,
                mlp_intermediate_dim=layer_config.mlp_intermediate_dim
            )
            for layer_config in config.layer_configs
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CompactedViTModel(nn.Module):
    def __init__(self, config: CompactedViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = CompactedViTEmbeddings(config)
        self.encoder = CompactedViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embeddings(pixel_values)
        encoder_output = self.encoder(embedding_output)
        return self.layernorm(encoder_output)


class CompactedViTForImageClassification(nn.Module):
    def __init__(self, config: CompactedViTConfig):
        super().__init__()
        self.config = config
        self.vit = CompactedViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        sequence_output = self.vit(pixel_values)
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}


# ==============================================================================
# LOAD FUNCTION
# ==============================================================================

def load_compacted_model(
    load_path: str,
    device: str = 'cpu'
) -> Tuple[CompactedViTForImageClassification, Dict]:
    """
    Load a compacted ViT model saved by save_compacted_model() in the notebook.
    The file is a dict with keys: state_dict, config, model_type, [keep_masks], [metadata].
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    config = CompactedViTConfig.from_dict(checkpoint['config'])
    model = CompactedViTForImageClassification(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    metadata = {
        'config': config,
        'keep_masks': checkpoint.get('keep_masks', None),
        'metadata': checkpoint.get('metadata', None),
    }
    return model, metadata

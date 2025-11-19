import torch
from typing_extensions import TypedDict


class CacheItem(TypedDict):
    num_tokens: int


class DataItem(CacheItem):
    input_ids: list[int]
    labels: list[int]


class BaseMLLMDataItem(DataItem):
    num_img_tokens: list[int]
    num_imgs: list[int]
    num_patches: list[int]


class InternS1DataItem(BaseMLLMDataItem):
    pixel_values: torch.Tensor
    image_flags: torch.Tensor


class QwenVL3DataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor


class OmniDataItem(BaseMLLMDataItem, total=False):
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor
    image_flags: torch.Tensor

class Qwen25OmniDataItem(BaseMLLMDataItem, total=False):
    """
    Data item for Qwen2.5-Omni model supporting vision and audio input.
    
    Inherits from BaseMLLMDataItem:
        - input_ids: torch.Tensor
        - labels: torch.Tensor
        - attention_mask: Optional[torch.Tensor]
    
    Additional fields:
        Vision: pixel_values, image_grid_thw, image_flags
        Audio: input_features, audio_position_ids
        Position: position_ids (for multi-modal position encoding)
    """
    
    # Vision fields
    pixel_values: torch.Tensor  # [B, C, H, W] or [B, T, C, H, W]
    image_grid_thw: torch.Tensor  # [num_images, 3] - (temporal, height, width)
    image_flags: torch.Tensor  # [num_images] - 0: image, 1: video
    
    # Audio fields
    input_features: torch.Tensor  # [B, num_mel_bins, time_steps]
    audio_position_ids: torch.Tensor  # Position IDs for audio tokens
    
    # Position encoding
    position_ids: torch.Tensor  # Multi-modal position IDs
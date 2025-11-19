import torch
import torch.nn as nn
from typing import Optional
from functools import partial
from pathlib import Path

from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils import get_device, get_torch_device_module, init_params, get_logger
from xtuner.v1.ops.act_fn import get_act_fn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from typing_extensions import override

from .qwen2_5_omni_config import Qwen2_5OmniProjectorConfig
from .utils import init_world_mesh

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class Qwen2_5OmniVisionProjector(nn.Module):
    """Vision projection module that merges vision patches and projects to text hidden size"""
    
    def __init__(self, config: Qwen2_5OmniProjectorConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size * (config.spatial_merge_size ** 2)
        self.config = config
        
        self.ln_q = nn.LayerNorm(config.vision_hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.text_hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vision features [batch, num_patches, vision_hidden_size]
        
        Returns:
            Projected features [batch, num_merged_patches, text_hidden_size]
        """
        # Apply layer norm first
        x = self.ln_q(x)
        
        # Merge spatial patches
        # Reshape to merge spatial_merge_size x spatial_merge_size patches
        # x shape: [batch, num_patches, vision_hidden_size]
        # After merge: [batch, num_merged_patches, vision_hidden_size * spatial_merge_size^2]
        x = x.view(-1, self.hidden_size)
        
        # Project to text hidden size
        x = self.mlp(x)
        return x


class Qwen2_5OmniAudioProjector(nn.Module):
    """Audio projection module that projects audio features to text hidden size"""
    
    def __init__(self, config: Qwen2_5OmniProjectorConfig):
        super().__init__()
        self.config = config
        # Audio features are already processed by audio encoder
        # Just need a simple projection if dimensions don't match
        if config.audio_hidden_size != config.text_hidden_size:
            self.proj = nn.Linear(config.audio_hidden_size, config.text_hidden_size)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio features [batch, num_frames, audio_hidden_size]
        
        Returns:
            Projected features [batch, num_frames, text_hidden_size]
        """
        return self.proj(x)


class Qwen2_5OmniProjector(BaseModel):
    """
    Multi-modal projector for Qwen2.5-Omni that projects vision and audio features 
    to the text model's hidden size.
    """
    config: Qwen2_5OmniProjectorConfig
    
    def __init__(self, config: Qwen2_5OmniProjectorConfig):
        super().__init__()
        self.config = config
        
        self.vision_merger = Qwen2_5OmniVisionProjector(config)
        self.audio_proj = Qwen2_5OmniAudioProjector(config)
        
        self._hf_prefix = "multi_modal_projector."
        self._init_load_spec()
    
    @maybe_compile(fullgraph=True)
    def forward(
        self, 
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Project vision and audio features to text hidden size.
        
        Args:
            vision_features: Vision features from vision encoder 
                            [batch, num_patches, vision_hidden_size]
            audio_features: Audio features from audio encoder
                           [batch, num_frames, audio_hidden_size]
        
        Returns:
            Tuple of (projected_vision_features, projected_audio_features)
            - projected_vision_features: [batch, num_merged_patches, text_hidden_size]
            - projected_audio_features: [batch, num_frames, text_hidden_size]
        """
        projected_vision = None
        projected_audio = None
        
        if vision_features is not None:
            projected_vision = self.vision_merger(vision_features)
        
        if audio_features is not None:
            projected_audio = self.audio_proj(audio_features)
        
        return projected_vision, projected_audio
    
    def to_hf_key_list(self, key: str) -> list[str]:
        """Convert internal key to HuggingFace checkpoint key(s)"""
        return [self._hf_prefix + key]
    
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Optional[Float8Handler] = None,
    ):
        """Apply FSDP to the projector"""
        self.fsdp_config = fsdp_config
        assert float8_handler is None, "Float8 not supported for projector"
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype
        )
        
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None
        
        self._maybe_compile_layers()
        
        # Convert parameters to fp32 if training
        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            # Freeze all parameters if not training
            for param in self.parameters():
                param.requires_grad = False
        
        # Apply FSDP
        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        
        self._to_empty_meta()
        return self
    
    @torch.no_grad()
    def init_weights(self):
        """Initialize all weights of the projector"""
        initialized_params = set()
        
        # Initialize vision merger
        # LayerNorm
        init_params(self.vision_merger.ln_q.weight, nn.init.ones_)
        initialized_params.add("vision_merger.ln_q.weight")
        init_params(self.vision_merger.ln_q.bias, nn.init.zeros_)
        initialized_params.add("vision_merger.ln_q.bias")
        
        # MLP layers
        for idx, layer in enumerate(self.vision_merger.mlp):
            if isinstance(layer, nn.Linear):
                init_params(layer.weight,
                           partial(nn.init.normal_, mean=0.0, std=0.02))
                initialized_params.add(f"vision_merger.mlp.{idx}.weight")
                if layer.bias is not None:
                    init_params(layer.bias, nn.init.zeros_)
                    initialized_params.add(f"vision_merger.mlp.{idx}.bias")
        
        # Initialize audio projector (if not Identity)
        if isinstance(self.audio_proj.proj, nn.Linear):
            init_params(self.audio_proj.proj.weight,
                       partial(nn.init.normal_, mean=0.0, std=0.02))
            initialized_params.add("audio_proj.proj.weight")
            if self.audio_proj.proj.bias is not None:
                init_params(self.audio_proj.proj.bias, nn.init.zeros_)
                initialized_params.add("audio_proj.proj.bias")
        
        # Verify all parameters are initialized
        expected_param_name = {self._clean_param_name(name) for name, _ in self.named_parameters()}
        if missing := expected_param_name - initialized_params:
            raise RuntimeError(f"{missing} is not initialized")
        
        logger.info(f"Projector initialized {len(initialized_params)} parameters")
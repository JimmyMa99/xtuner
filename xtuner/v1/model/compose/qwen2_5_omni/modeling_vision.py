import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import partial

from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils import get_device, get_torch_device_module, init_params
from xtuner.v1.ops.attn_imp import attn_impl_mapping
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from .qwen2_5_omni_config import Qwen2_5OmniVisionConfig
from .utils import init_world_mesh

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class Qwen2_5OmniVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2_5OmniVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, 
            self.embed_dim, 
            kernel_size=kernel_size, 
            stride=kernel_size, 
            bias=False
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5OmniVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        
        self.q = nn.Linear(self.dim, self.dim, bias=True)
        self.k = nn.Linear(self.dim, self.dim, bias=True)
        self.v = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.attn_impl_func = attn_impl_mapping[config.attn_impl]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):
        seq_length = hidden_states.shape[0]
        
        query_states = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        
        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
            key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        
        attn_output = self.attn_impl_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=False,
        )
        
        attn_output = attn_output[0].reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionMLP(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionConfig):
        super().__init__()
        from xtuner.v1.ops.act_fn import get_act_fn
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = get_act_fn(config.hidden_act)
    
    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniVisionBlock(nn.Module):
    def __init__(self, config: Qwen2_5OmniVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5OmniVisionAttention(config)
        self.mlp = Qwen2_5OmniVisionMLP(config)
    
    @maybe_compile(fullgraph=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5OmniVisionEncoder(BaseModel):
    config: Qwen2_5OmniVisionConfig
    
    def __init__(self, config: Qwen2_5OmniVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        
        self.patch_embed = Qwen2_5OmniVisionPatchEmbed(config)
        
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = Qwen2_5OmniVisionRotaryEmbedding(head_dim // 2)
        
        self.blocks = nn.ModuleList([
            Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)
        ])
        
        self._hf_prefix = "visual."
        self._init_load_spec()
    
    def rot_pos_emb(self, grid_thw):
        # 实现位置编码逻辑
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
    
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_pos_emb=rotary_pos_emb
            )
        
        return hidden_states
    
    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]
    
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Optional[Float8Handler] = None,
    ):
        self.fsdp_config = fsdp_config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype
        )
        
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None
        
        self._maybe_compile_layers()
        
        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False
        
        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self
    
    @torch.no_grad()
    def init_weights(self):
        # 初始化权重
        initialized_params = set()
        
        init_params(self.patch_embed.proj.weight, 
                   partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
        initialized_params.add("patch_embed.proj.weight")
        
        for layer_idx, layer in enumerate(self.blocks):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    init_params(module.weight,
                              partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
                    initialized_params.add(f"blocks.{layer_idx}.{name}.weight")
                    if module.bias is not None:
                        init_params(module.bias, nn.init.zeros_)
                        initialized_params.add(f"blocks.{layer_idx}.{name}.bias")
                elif isinstance(module, nn.LayerNorm):
                    init_params(module.weight, nn.init.ones_)
                    init_params(module.bias, nn.init.zeros_)
                    initialized_params.add(f"blocks.{layer_idx}.{name}.weight")
                    initialized_params.add(f"blocks.{layer_idx}.{name}.bias")
        
        expected_param_name = {self._clean_param_name(name) for name, _ in self.named_parameters()}
        if missing := expected_param_name - initialized_params:
            raise RuntimeError(f"{missing} is not initialized")
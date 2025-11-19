import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from functools import partial
from pathlib import Path

from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.ops.attn_imp import attn_impl_mapping
from xtuner.v1.utils import get_device, get_logger, init_params, XTUNER_DETERMINISTIC
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from typing_extensions import override
from xtuner.v1.ops.act_fn import get_act_fn
from tqdm import tqdm

from .qwen2_5_omni_config import Qwen2_5OmniAudioConfig
from .utils import init_world_mesh

DEVICE = get_device()
logger = get_logger()


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )
    
    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention for audio encoder"""
    
    def __init__(self, config: Qwen2_5OmniAudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config
        
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_decoder = False
        self.is_causal = False
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_impl_func = attn_impl_mapping[config.attn_impl]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Input shape: Seq_len x Channel"""
        seq_length, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        
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
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scaling,
            causal=False,
            deterministic=XTUNER_DETERMINISTIC,
        )
        
        attn_output = attn_output[0].reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(config.activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: input to the layer of shape (seq_len, embed_dim)
            cu_seqlens: cumulative sequence lengths
            max_seqlen: maximum sequence length
            attention_mask: attention mask
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        return hidden_states


class Qwen2_5OmniAudioEncoder(BaseModel):
    config: Qwen2_5OmniAudioConfig
    
    def __init__(self, config: Qwen2_5OmniAudioConfig):
        super().__init__()
        self.config = config
        
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.dropout = config.dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # Positional embedding
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        
        # Audio BOS/EOS token embedding
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        
        # Post-processing layers
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        
        self._hf_prefix = "audio_tower."
        self._init_load_spec()
    
    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepare attention mask for non-flash attention implementations"""
        if self.config.attn_impl == "flash_attention":
            return None
        
        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask
    
    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        aftercnn_lens: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_features: input mel spectrogram features
            feature_lens: mel length for each sample
            aftercnn_lens: mel length after CNN processing for each sample
        """
        # Process audio in chunks
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)
        
        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self._padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0
        )
        
        # Apply convolutional layers
        padded_embed = F.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = F.gelu(self.conv2(padded_embed)).transpose(1, 2)
        
        # Add positional embeddings
        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        
        hidden_states = padded_embed[padded_mask_after_cnn]
        
        # Prepare cumulative sequence lengths for attention
        cu_seqlens = torch.cat([
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0)
        ]).to(torch.int32)
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)
        
        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=attention_mask,
            )
        
        # Post-processing
        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        
        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio
    
    def _padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )
        
        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]
        
        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths
    
    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]
    
    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict)
        # Rebuild positional embedding if loaded from meta device
        embed_dim = self.config.d_model
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        return loaded_keys, unloaded_keys, missing_keys
    
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Optional[Float8Handler] = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None, "Float8 not supported for audio encoder"
        
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
        
        # Rebuild positional embedding after FSDP
        embed_dim = self.config.d_model
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        
        # Apply FSDP to each layer
        for layer_idx in tqdm(list(range(len(self.layers))), desc="[Audio Encoder Fully Shard]"):
            layer = self.layers[layer_idx]
            
            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )
        
        # Set forward prefetch
        for layer_cur, layer_next in zip(self.layers[:-1], self.layers[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])
        
        # Apply FSDP to the whole module
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
        """Initialize all weights of the audio encoder"""
        initialized_params = set()
        
        # Initialize conv1
        if hasattr(self.conv1, 'weight'):
            init_params(self.conv1.weight, 
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add("conv1.weight")
        if hasattr(self.conv1, 'bias') and self.conv1.bias is not None:
            init_params(self.conv1.bias, nn.init.zeros_)
            initialized_params.add("conv1.bias")
        
        # Initialize conv2
        if hasattr(self.conv2, 'weight'):
            init_params(self.conv2.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add("conv2.weight")
        if hasattr(self.conv2, 'bias') and self.conv2.bias is not None:
            init_params(self.conv2.bias, nn.init.zeros_)
            initialized_params.add("conv2.bias")
        
        # Initialize audio_bos_eos_token embedding
        init_params(self.audio_bos_eos_token.weight,
                   partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
        initialized_params.add("audio_bos_eos_token.weight")
        
        # Initialize transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention LayerNorm
            init_params(layer.self_attn_layer_norm.weight, nn.init.ones_)
            initialized_params.add(f"layers.{layer_idx}.self_attn_layer_norm.weight")
            init_params(layer.self_attn_layer_norm.bias, nn.init.zeros_)
            initialized_params.add(f"layers.{layer_idx}.self_attn_layer_norm.bias")
            
            # Self-attention projections
            # Q projection
            init_params(layer.self_attn.q_proj.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.self_attn.q_proj.weight")
            if layer.self_attn.q_proj.bias is not None:
                init_params(layer.self_attn.q_proj.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.self_attn.q_proj.bias")
            
            # K projection
            init_params(layer.self_attn.k_proj.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.self_attn.k_proj.weight")
            if layer.self_attn.k_proj.bias is not None:
                init_params(layer.self_attn.k_proj.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.self_attn.k_proj.bias")
            
            # V projection
            init_params(layer.self_attn.v_proj.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.self_attn.v_proj.weight")
            if layer.self_attn.v_proj.bias is not None:
                init_params(layer.self_attn.v_proj.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.self_attn.v_proj.bias")
            
            # Output projection
            init_params(layer.self_attn.out_proj.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.self_attn.out_proj.weight")
            if layer.self_attn.out_proj.bias is not None:
                init_params(layer.self_attn.out_proj.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.self_attn.out_proj.bias")
            
            # Final LayerNorm
            init_params(layer.final_layer_norm.weight, nn.init.ones_)
            initialized_params.add(f"layers.{layer_idx}.final_layer_norm.weight")
            init_params(layer.final_layer_norm.bias, nn.init.zeros_)
            initialized_params.add(f"layers.{layer_idx}.final_layer_norm.bias")
            
            # FFN fc1
            init_params(layer.fc1.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.fc1.weight")
            if layer.fc1.bias is not None:
                init_params(layer.fc1.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.fc1.bias")
            
            # FFN fc2
            init_params(layer.fc2.weight,
                       partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
            initialized_params.add(f"layers.{layer_idx}.fc2.weight")
            if layer.fc2.bias is not None:
                init_params(layer.fc2.bias, nn.init.zeros_)
                initialized_params.add(f"layers.{layer_idx}.fc2.bias")
        
        # Initialize ln_post (post LayerNorm)
        init_params(self.ln_post.weight, nn.init.ones_)
        initialized_params.add("ln_post.weight")
        init_params(self.ln_post.bias, nn.init.zeros_)
        initialized_params.add("ln_post.bias")
        
        # Initialize projection layer
        init_params(self.proj.weight,
                   partial(nn.init.normal_, mean=0.0, std=self.config.initializer_range))
        initialized_params.add("proj.weight")
        if self.proj.bias is not None:
            init_params(self.proj.bias, nn.init.zeros_)
            initialized_params.add("proj.bias")
        
        # Verify all parameters are initialized
        expected_param_name = {self._clean_param_name(name) for name, _ in self.named_parameters()}
        
        # Remove persistent buffers from expected params (like positional_embedding.positional_embedding)
        expected_param_name = {name for name in expected_param_name 
                              if not name.startswith("positional_embedding.")}
        
        if missing := expected_param_name - initialized_params:
            raise RuntimeError(f"{missing} is not initialized")
        
        logger.info(f"Audio encoder initialized {len(initialized_params)} parameters")
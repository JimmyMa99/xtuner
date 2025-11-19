from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from mmengine import is_installed
from xtuner.v1.float8 import Float8Config
from xtuner.v1.utils import get_device, get_logger

logger = get_logger()


class Qwen2_5OmniProjectorConfig(BaseModel):
    """Projector configuration for Qwen2.5-Omni"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Projector Config",
        extra="forbid",
    )
    
    vision_hidden_size: int = 1280
    audio_hidden_size: int = 1280
    text_hidden_size: int = 2048  # 3B: 2048, 7B: 3584
    spatial_merge_size: int = 2
    
    def build(self):
        from .modeling_projector import Qwen2_5OmniProjector
        return Qwen2_5OmniProjector(self)


class Qwen2_5OmniVisionConfig(BaseModel):
    """Vision encoder configuration for Qwen2.5-Omni"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Vision Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_vision_encoder"
    in_channels: int = 3
    depth: int = 32
    hidden_size: int = 1280
    num_attention_heads: int = 16
    intermediate_size: int = 3420
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    out_hidden_size: int = 2048  # 3B: 2048, 7B: 3584
    fullatt_block_indexes: List[int] = [7, 15, 23, 31]
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    float8_cfg: Optional[Float8Config] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"
    
    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self
    
    def build(self):
        from .modeling_vision import Qwen2_5OmniVisionEncoder
        return Qwen2_5OmniVisionEncoder(self)


class Qwen2_5OmniAudioConfig(BaseModel):
    """Audio encoder configuration for Qwen2.5-Omni"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Audio Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_audio_encoder"
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 100
    output_dim: int = 2048  # 3B: 2048, 7B: 3584
    float8_cfg: Optional[Float8Config] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"
    
    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self
    
    def build(self):
        from .modeling_audio import Qwen2_5OmniAudioEncoder
        return Qwen2_5OmniAudioEncoder(self)


class Qwen2_5OmniTextConfig(BaseModel):
    """Text model configuration for Qwen2.5-Omni"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Text Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_text"
    
    # Basic architecture
    vocab_size: int = 151936  # 3B: 151936, 7B: 152064
    hidden_size: int = 2048  # 3B: 2048, 7B: 3584
    intermediate_size: int = 11008  # 3B: 11008, 7B: 18944
    num_hidden_layers: int = 36  # 3B: 36, 7B: 28
    num_attention_heads: int = 16  # 3B: 16, 7B: 28
    num_key_value_heads: int = 2  # 3B: 2, 7B: 4
    
    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default"
        }
    )
    
    # Attention
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 70  # 3B: 70, 7B: 28
    
    # Other
    tie_word_embeddings: bool = False
    use_cache: bool = True
    
    def build(self):
        from .modeling_qwen2_5_omni_text import Qwen2_5OmniTextModel
        return Qwen2_5OmniTextModel(self)


class Qwen2_5OmniTalkerConfig(BaseModel):
    """Talker configuration for audio output"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Talker Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_talker"
    
    # Architecture
    vocab_size: int = 8448
    hidden_size: int = 896
    intermediate_size: int = 4864  # 3B: 4864, 7B: 18944
    num_hidden_layers: int = 24
    num_attention_heads: int = 14  # 3B: 14, 7B: 12
    num_key_value_heads: int = 2  # 3B: 2, 7B: 4
    head_dim: int = 64  # 3B: 64, 7B: 128
    embedding_size: int = 2048  # 3B: 2048, 7B: 3584
    
    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "mrope_section": [16, 16, 0],  # 3B: [16, 16, 0], 7B: [16, 24, 24]
            "rope_type": "default",
            "type": "default"
        }
    )
    
    # Attention
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 28
    spatial_merge_size: int = 2
    
    # Special tokens
    audio_token_index: int = 151646
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    image_token_index: int = 151655
    video_token_index: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    # TTS tokens
    tts_codec_start_token_id: int = 8293
    tts_codec_end_token_id: int = 8294
    tts_codec_pad_token_id: int = 8292
    tts_codec_mask_token_id: int = 8296
    tts_text_start_token_id: int = 151860
    tts_text_end_token_id: int = 151861
    tts_text_pad_token_id: int = 151859
    
    # Audio specific
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    
    # Other
    use_cache: bool = True
    
    def build(self):
        from .modeling_talker import Qwen2_5OmniTalker
        return Qwen2_5OmniTalker(self)


class Qwen2_5OmniThinkerConfig(BaseModel):
    """Thinker configuration (main multimodal model)"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Thinker Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_thinker"
    
    vision_config: Qwen2_5OmniVisionConfig
    audio_config: Qwen2_5OmniAudioConfig
    text_config: Qwen2_5OmniTextConfig
    
    # Special tokens
    audio_token_index: int = 151646
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    image_token_index: int = 151655
    video_token_index: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    
    bos_token_id: int = 151644
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    user_token_id: int = 872
    
    # Audio specific
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    
    # Training
    ignore_index: int = -100
    initializer_range: float = 0.02
    
    def build(self):
        from .modeling_thinker import Qwen2_5OmniThinker
        return Qwen2_5OmniThinker(self)


class Qwen2_5OmniBigVGANConfig(BaseModel):
    """BigVGAN vocoder configuration"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni BigVGAN Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_bigvgan"
    
    mel_dim: int = 80
    upsample_initial_channel: int = 1536
    upsample_rates: List[int] = [5, 3, 2, 2, 2, 2]
    upsample_kernel_sizes: List[int] = [11, 7, 4, 4, 4, 4]
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    use_bias_at_final: bool = False
    
    def build(self):
        from .modeling_bigvgan import Qwen2_5OmniBigVGAN
        return Qwen2_5OmniBigVGAN(self)


class Qwen2_5OmniDiTConfig(BaseModel):
    """DiT (Diffusion Transformer) configuration"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni DiT Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_dit"
    
    # Architecture
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    head_dim: int = 64
    ff_mult: int = 2
    dropout: float = 0.1
    
    # Embeddings
    num_embeds: int = 8193
    emb_dim: int = 512
    mel_dim: int = 80
    
    # Encoder config
    enc_dim: int = 128
    enc_emb_dim: int = 192
    enc_channels: List[int] = [256, 256, 256, 256, 768]
    enc_kernel_sizes: List[int] = [5, 3, 3, 3, 1]
    enc_dilations: List[int] = [1, 2, 3, 4, 1]
    enc_attention_channels: int = 64
    enc_se_channels: int = 64
    enc_lin_neurons: int = 192
    enc_res2net_scale: int = 2
    enc_global_context: bool = True
    
    repeats: int = 2
    
    def build(self):
        from .modeling_dit import Qwen2_5OmniDiT
        return Qwen2_5OmniDiT(self)


class Qwen2_5OmniToken2WavConfig(BaseModel):
    """Token2Wav configuration"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Token2Wav Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni_token2wav"
    
    bigvgan_config: Qwen2_5OmniBigVGANConfig = Field(default_factory=Qwen2_5OmniBigVGANConfig)
    dit_config: Qwen2_5OmniDiTConfig = Field(default_factory=Qwen2_5OmniDiTConfig)
    
    def build(self):
        from .modeling_token2wav import Qwen2_5OmniToken2Wav
        return Qwen2_5OmniToken2Wav(self)


class Qwen2_5OmniBaseConfig(BaseModel):
    """Base configuration for Qwen2.5-Omni models"""
    model_config = ConfigDict(
        title="Qwen2.5 Omni Base Config",
        extra="forbid",
    )
    
    model_type: str = "qwen2_5_omni"
    
    thinker_config: Qwen2_5OmniThinkerConfig
    talker_config: Optional[Qwen2_5OmniTalkerConfig] = None
    token2wav_config: Optional[Qwen2_5OmniToken2WavConfig] = None
    projector_config: Optional[Qwen2_5OmniProjectorConfig] = None
    
    # Audio output control
    enable_audio_output: bool = True
    enable_talker: bool = True
    
    # Freeze options
    freeze_vision: bool = False
    freeze_audio: bool = False
    freeze_text: bool = False
    freeze_talker: bool = False
    freeze_projector: bool = False
    
    # Training and saving options
    hf_save_worker: int = 16
    dcp_ignore_frozen_params: bool = True
    
    def build(self):
        from .modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        return Qwen2_5OmniForConditionalGeneration(self)
    
    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError
    
    @property
    def hf_config(self):
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None


class Qwen2_5Omni3BConfig(Qwen2_5OmniBaseConfig):
    """Qwen2.5-Omni 3B model configuration"""
    
    thinker_config: Qwen2_5OmniThinkerConfig = Field(
        default_factory=lambda: Qwen2_5OmniThinkerConfig(
            vision_config=Qwen2_5OmniVisionConfig(
                out_hidden_size=2048,
            ),
            audio_config=Qwen2_5OmniAudioConfig(
                output_dim=2048,
            ),
            text_config=Qwen2_5OmniTextConfig(
                vocab_size=151936,
                hidden_size=2048,
                intermediate_size=11008,
                num_hidden_layers=36,
                num_attention_heads=16,
                num_key_value_heads=2,
                max_window_layers=70,
                rope_scaling={
                    "mrope_section": [16, 24, 24],
                    "rope_type": "default",
                    "type": "default"
                },
            ),
        )
    )
    
    talker_config: Qwen2_5OmniTalkerConfig = Field(
        default_factory=lambda: Qwen2_5OmniTalkerConfig(
            embedding_size=2048,
            intermediate_size=4864,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            rope_scaling={
                "mrope_section": [16, 16, 0],
                "rope_type": "default",
                "type": "default"
            },
        )
    )
    
    token2wav_config: Qwen2_5OmniToken2WavConfig = Field(
        default_factory=Qwen2_5OmniToken2WavConfig
    )
    
    projector_config: Qwen2_5OmniProjectorConfig = Field(
        default_factory=lambda: Qwen2_5OmniProjectorConfig(
            vision_hidden_size=1280,
            audio_hidden_size=1280,
            text_hidden_size=2048,
            spatial_merge_size=2,
        )
    )


class Qwen2_5Omni7BConfig(Qwen2_5OmniBaseConfig):
    """Qwen2.5-Omni 7B model configuration"""
    
    thinker_config: Qwen2_5OmniThinkerConfig = Field(
        default_factory=lambda: Qwen2_5OmniThinkerConfig(
            vision_config=Qwen2_5OmniVisionConfig(
                out_hidden_size=3584,
            ),
            audio_config=Qwen2_5OmniAudioConfig(
                output_dim=3584,
            ),
            text_config=Qwen2_5OmniTextConfig(
                vocab_size=152064,
                hidden_size=3584,
                intermediate_size=18944,
                num_hidden_layers=28,
                num_attention_heads=28,
                num_key_value_heads=4,
                max_window_layers=28,
                rope_scaling={
                    "mrope_section": [16, 24, 24],
                    "rope_type": "default",
                    "type": "default"
                },
            ),
        )
    )
    
    talker_config: Qwen2_5OmniTalkerConfig = Field(
        default_factory=lambda: Qwen2_5OmniTalkerConfig(
            embedding_size=3584,
            intermediate_size=18944,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=128,
            rope_scaling={
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default"
            },
        )
    )
    
    token2wav_config: Qwen2_5OmniToken2WavConfig = Field(
        default_factory=Qwen2_5OmniToken2WavConfig
    )
    
    projector_config: Qwen2_5OmniProjectorConfig = Field(
        default_factory=lambda: Qwen2_5OmniProjectorConfig(
            vision_hidden_size=1280,
            audio_hidden_size=1280,
            text_hidden_size=3584,
            spatial_merge_size=2,
        )
    )


# Alias for backward compatibility
Qwen2_5OmniConfig = Qwen2_5Omni3BConfig
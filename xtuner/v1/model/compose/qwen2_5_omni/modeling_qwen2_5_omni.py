import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from typing_extensions import override

from xtuner.v1.model import BaseModel
from xtuner.v1.loss import CELossContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import get_logger
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from .qwen2_5_omni_config import Qwen2_5OmniConfig
from .modeling_vision import Qwen2_5OmniVisionEncoder
from .modeling_audio import Qwen2_5OmniAudioEncoder
from .modeling_projector import Qwen2_5OmniProjector
from .utils import init_world_mesh

logger = get_logger()


class Qwen2_5OmniForConditionalGeneration(BaseModel):
    config: Qwen2_5OmniConfig
    
    def __init__(self, config: Qwen2_5OmniConfig):
        super().__init__()
        self.config = config
        
        # Initialize sub-modules
        self.visual = config.vision_config.build()
        self.audio_tower = config.audio_config.build()
        self.multi_modal_projector = config.projector_config.build()
        self.language_model = config.text_config.build()
        
        self._hf_path: Optional[Path] = None
        
        # Build global load spec mapping for save_hf
        self.load_spec_mapping = {}
        for key, value in self.visual.load_spec_mapping.items():
            self.load_spec_mapping['visual.' + key] = value
        for key, value in self.audio_tower.load_spec_mapping.items():
            self.load_spec_mapping['audio_tower.' + key] = value
        for key, value in self.multi_modal_projector.load_spec_mapping.items():
            self.load_spec_mapping['multi_modal_projector.' + key] = value
        for key, value in self.language_model.load_spec_mapping.items():
            self.load_spec_mapping['language_model.' + key] = value
        
        self._freeze_modules()
    
    def _freeze_modules(self):
        """Freeze modules based on config"""
        if self.config.freeze_vision:
            self.visual.requires_grad_(False)
            self.visual.eval()
            logger.info("Freeze vision encoder")
        
        if self.config.freeze_audio:
            self.audio_tower.requires_grad_(False)
            self.audio_tower.eval()
            logger.info("Freeze audio encoder")
        
        if self.config.freeze_projector:
            self.multi_modal_projector.requires_grad_(False)
            self.multi_modal_projector.eval()
            logger.info("Freeze multi modal projector")
        
        if self.config.freeze_language:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
            logger.info("Freeze language model")
    
    @override
    def init_weights(self) -> None:
        """Initialize weights for all sub-modules"""
        self.visual.init_weights()
        self.audio_tower.init_weights()
        self.multi_modal_projector.init_weights()
        self.language_model.init_weights()
        logger.info("All sub-modules initialized")
    
    def get_visual_features(
        self, 
        pixel_values: torch.Tensor, 
        grid_thw: torch.LongTensor
    ) -> torch.Tensor:
        """
        Encodes images/videos into continuous embeddings.
        
        Args:
            pixel_values: The tensors corresponding to the input images/videos
            grid_thw: The temporal, height and width of feature shape of each image/video
        
        Returns:
            Projected visual embeddings
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        visual_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        
        # Project to text hidden size
        visual_embeds, _ = self.multi_modal_projector(vision_features=visual_embeds)
        
        # Split by grid_thw
        split_sizes = (grid_thw.prod(-1) // self.config.vision_config.spatial_merge_size ** 2).tolist()
        visual_embeds = torch.split(visual_embeds, split_sizes)
        
        return visual_embeds
    
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Encodes audios into continuous embeddings.
        
        Args:
            input_features: The tensors corresponding to the input audios (mel spectrograms)
            feature_attention_mask: Mask to avoid performing attention on padding feature indices
        
        Returns:
            Projected audio embeddings
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = feature_attention_mask.sum(-1) if feature_attention_mask is not None else None
        
        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        
        audio_features = self.audio_tower(
            input_features=input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )
        
        # Project to text hidden size
        _, audio_features = self.multi_modal_projector(audio_features=audio_features)
        
        return audio_features
    
    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from input_ids or inputs_embeds.
        Checks that placeholder token count equals the length of multimodal features.
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id
        
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_features.shape[0]}"
            )
        
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                f"features {video_features.shape[0]}"
            )
        
        n_audio_tokens = special_audio_mask.sum()
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if audio_features is not None and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
            raise ValueError(
                f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, "
                f"features {audio_features.shape[0]}"
            )
        
        return special_image_mask, special_video_mask, special_audio_mask
    
    def forward(
        self,
        seq_ctx: SequenceContext,
        loss_ctx: CELossContext
    ):
        """
        Forward pass for the full multimodal model.
        
        Args:
            seq_ctx: Sequence context containing input_ids, pixel_values, input_features, etc.
            loss_ctx: Loss context
        
        Returns:
            Model outputs including loss and logits
        """
        input_ids = seq_ctx.input_ids
        pixel_values = seq_ctx.pixel_values
        pixel_values_videos = seq_ctx.pixel_values_videos
        image_grid_thw = seq_ctx.image_grid_thw
        video_grid_thw = seq_ctx.video_grid_thw
        input_features = seq_ctx.input_features
        feature_attention_mask = seq_ctx.feature_attention_mask
        
        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        
        # Process images if present
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.get_visual_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        # Process videos if present
        if pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds = self.get_visual_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        
        # Process audio if present
        if input_features is not None:
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        
        # Create language model sequence context
        lang_seq_ctx = SequenceContext(
            input_ids=None,
            cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
            cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
            max_length_q=seq_ctx.max_length_q,
            max_length_k=seq_ctx.max_length_k,
            position_ids=seq_ctx.position_ids,
            num_padding=seq_ctx.num_padding,
            sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
            inputs_embeds=inputs_embeds,
        )
        
        # Forward through language model
        outputs = self.language_model(lang_seq_ctx, loss_ctx)
        
        return outputs
    
    def scale_and_reduce_grad(self):
        """Scale and reduce gradients (for MoE models)"""
        self.language_model.scale_and_reduce_grad()
    
    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True):
        """Load model from HuggingFace checkpoint"""
        self._hf_path = Path(hf_path)
        
        if isinstance(hf_path, Path):
            hf_path = str(hf_path)
        
        _, _, missing_llm_keys = self.language_model.from_hf(hf_path, strict=False)
        _, _, missing_vision_keys = self.visual.from_hf(hf_path, strict=False)
        _, _, missing_audio_keys = self.audio_tower.from_hf(hf_path, strict=False)
        _, _, missing_projector_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)
        
        missing = missing_llm_keys | missing_vision_keys | missing_audio_keys | missing_projector_keys
        if strict and missing:
            raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}")
        
        return set(), set(), missing
    
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Optional[Float8Handler] = None,
    ):
        """Apply FSDP to the model"""
        self.fsdp_config = fsdp_config
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype
        )
        
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None
        
        # Apply FSDP to the full model
        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        
        self._to_empty_meta()
        return self
    
    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        """Convert model parameter to safetensor format for HuggingFace"""
        # Delegate to language model if it's a MoE model with special weight layouts
        if hasattr(self.language_model, 'param_to_safetensor'):
            # Check if this is a language model parameter
            if hf_param_name.startswith('model.'):
                return self.language_model.param_to_safetensor(safetensor, hf_param_name)
        
        return safetensor
    
    def to_hf_key_list(self, key: str) -> list[str]:
        """Convert internal key to HuggingFace key format"""
        # This method should not be called directly as we use sub-modules' load_spec_mapping
        raise NotImplementedError(
            "to_hf_key_list should not be called on the main model. "
            "Use sub-modules' load_spec_mapping instead."
        )


class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniForConditionalGeneration):
    """
    Qwen2.5-Omni Thinker model - processes multimodal inputs and generates text.
    This is the main model that handles image, video, audio, and text inputs.
    """
    
    def __init__(self, config: Qwen2_5OmniConfig):
        super().__init__(config)
        logger.info("Initialized Qwen2.5-Omni Thinker model")
    
    def forward(
        self,
        seq_ctx: SequenceContext,
        loss_ctx: CELossContext
    ):
        """
        Forward pass for thinker model.
        
        Processes multimodal inputs (text, image, video, audio) and generates text tokens.
        """
        input_ids = seq_ctx.input_ids
        pixel_values = getattr(seq_ctx, 'pixel_values', None)
        pixel_values_videos = getattr(seq_ctx, 'pixel_values_videos', None)
        image_grid_thw = getattr(seq_ctx, 'image_grid_thw', None)
        video_grid_thw = getattr(seq_ctx, 'video_grid_thw', None)
        input_features = getattr(seq_ctx, 'input_features', None)
        feature_attention_mask = getattr(seq_ctx, 'feature_attention_mask', None)
        
        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        
        # Process images
        if pixel_values is not None and image_grid_thw is not None:
            try:
                image_embeds = self.get_visual_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            except Exception as e:
                logger.warning(f"Error processing images: {e}, creating dummy features")
                # Create dummy features
                dummy_image = torch.randn(4, inputs_embeds.shape[-1], 
                                         device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + dummy_image.sum() * 0.0
        
        # Process videos
        if pixel_values_videos is not None and video_grid_thw is not None:
            try:
                video_embeds = self.get_visual_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            except Exception as e:
                logger.warning(f"Error processing videos: {e}, creating dummy features")
                dummy_video = torch.randn(4, inputs_embeds.shape[-1],
                                         device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + dummy_video.sum() * 0.0
        
        # Process audio
        if input_features is not None:
            try:
                audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
                audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                _, _, audio_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            except Exception as e:
                logger.warning(f"Error processing audio: {e}, creating dummy features")
                dummy_audio = torch.randn(4, inputs_embeds.shape[-1],
                                         device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                inputs_embeds = inputs_embeds + dummy_audio.sum() * 0.0
        
        # Create language model sequence context
        lang_seq_ctx = SequenceContext(
            input_ids=None,
            cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
            cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
            max_length_q=seq_ctx.max_length_q,
            max_length_k=seq_ctx.max_length_k,
            position_ids=seq_ctx.position_ids,
            num_padding=seq_ctx.num_padding,
            sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
            inputs_embeds=inputs_embeds,
        )
        
        # Forward through language model
        outputs = self.language_model(lang_seq_ctx, loss_ctx)
        
        return outputs
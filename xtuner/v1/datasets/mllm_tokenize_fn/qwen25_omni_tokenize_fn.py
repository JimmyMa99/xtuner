# Copyright (c) OpenMMLab. All rights reserved.

import hashlib
import os
import time
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP, HybridChatTemplate
from xtuner.v1.model import Qwen2_5OmniConfig
from xtuner.v1.utils import get_logger

from ..data_item import CacheItem, Qwen25OmniDataItem
from ..utils import apply_exif_orientation
from .base_mllm_tokenize_fn import (
    IMAGE_TOKEN_ALIAS,
    BaseMLLMTokenizeFnConfig,
    BaseMLLMTokenizeFunction,
    OSSLoaderConfig,
    get_image_path,
    load_image,
    replace_image_token,
)
from .qwen25_omni_process import (
    build_transform,
    dynamic_num_patch,
    dynamic_preprocess,
    process_audio,
)
from .qwen25_omni_utils import (
    Qwen25OmniOSSLoader,
    pil_loader,
    read_qwen25_omni_video,
)


logger = get_logger()

AUDIO_TOKEN_ALIAS = "<|AUDIO_PLACEHOLDER|>"
VIDEO_TOKEN_ALIAS = "<|VIDEO_PLACEHOLDER|>"


def dict_to_sorted_string(input_dict):
    """Convert a potentially nested dictionary into a sorted string representation."""

    def process_value(value):
        if isinstance(value, dict):
            return dict_to_sorted_string(value)
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    sorted_items = sorted((k, process_value(v)) for k, v in input_dict.items())
    return str(sorted_items)


def generate_random_int_from_dict(input_dict, min_num, max_num):
    """Generate a deterministic random integer based on a nested dictionary (using stable hashing)"""
    dict_string = dict_to_sorted_string(input_dict)
    input_bytes = dict_string.encode("utf-8")

    hash_hex = hashlib.md5(input_bytes).hexdigest()
    hash_int = int(hash_hex, 16)

    rng = np.random.default_rng(hash_int)
    return rng.integers(min_num, max_num + 1)


def replace_audio_token(messages: ChatMessages, chat_template: HybridChatTemplate, num_audio_token_list: list[int]):
    current_audio_idx = 0
    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        text = text.replace("<AUDIO_CONTEXT>", AUDIO_TOKEN_ALIAS)
                        audio_cnt = text.count(AUDIO_TOKEN_ALIAS)
                        for _ in range(audio_cnt):
                            audio_tokens = f"{chat_template.audio_start_token}{chat_template.audio_context_token * num_audio_token_list[current_audio_idx]}{chat_template.audio_end_token}"  # type: ignore
                            text = text.replace(AUDIO_TOKEN_ALIAS, audio_tokens, 1)
                            current_audio_idx += 1
                        c.text = text
    # if current_audio_idx < num_audio, it means <audio> placeholder is less than num_audio
    assert current_audio_idx == len(num_audio_token_list), (
        f"ERROR: current_audio_idx: {current_audio_idx} != num_audio: {len(num_audio_token_list)}"
    )


def replace_video_token(messages: ChatMessages, chat_template: HybridChatTemplate, num_image_token_list: list[int]):
    current_image_idx = 0
    n_frames = len(num_image_token_list)
    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        text = text.replace("<VIDEO_CONTEXT>", VIDEO_TOKEN_ALIAS)
                        video_cnt = text.count(VIDEO_TOKEN_ALIAS)
                        assert video_cnt == 1, "Only one <VIDEO_CONTEXT> is supported for video."
                        for _ in range(video_cnt):
                            special_tokens = "\n".join(
                                [f"Frame-{frame_idx + 1}: {IMAGE_TOKEN_ALIAS}" for frame_idx in range(n_frames)]
                            )
                            text = text.replace(VIDEO_TOKEN_ALIAS, special_tokens)
                            image_tokens = f"{chat_template.image_start_token}{chat_template.video_context_token * num_image_token_list[current_image_idx]}{chat_template.image_end_token}"  # type: ignore
                            text = text.replace(IMAGE_TOKEN_ALIAS, image_tokens)
                            current_image_idx += n_frames
                        c.text = text
    assert current_image_idx == len(num_image_token_list), (
        f"ERROR: current_image_idx: {current_image_idx} != num_image: {len(num_image_token_list)}"
    )


def replace_audio_in_video_token(messages: ChatMessages, chat_template: HybridChatTemplate, num_audio_token: int):
    """Replace <AUDIO_IN_VIDEO_CONTEXT> with actual audio tokens.
    
    Note: This uses the same audio_context_token as regular audio, following Swift's implementation.
    """
    for msg in messages.messages:
        if msg.role == "pretrain":
            assert len(messages.messages) == 1, "pretrain message should only have one message"
        if msg.role == "user" or msg.role == "pretrain":
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if c.type == "text":
                        text = c.text
                        if "<AUDIO_IN_VIDEO_CONTEXT>" in text:
                            # Use the same audio_context_token for audio in video
                            audio_tokens = f"{chat_template.audio_start_token}{chat_template.audio_context_token * num_audio_token}{chat_template.audio_end_token}"  # type: ignore
                            text = text.replace("<AUDIO_IN_VIDEO_CONTEXT>", audio_tokens)
                        c.text = text


class Qwen25OmniTokenizeFunction(BaseMLLMTokenizeFunction[Qwen25OmniDataItem]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor_path: str,
        anno_name: str,
        max_dynamic_patch: int | None = None,
        min_dynamic_patch: int | None = None,
        min_num_frames: int = 4,
        max_num_frames: int = 24,
        audio_sample_rate: int = 16000,
        audio_max_length: int = 30,
        data_augment: bool = False,
        system_message: str | None = None,
        oss_loader_cfg: OSSLoaderConfig | None = None,
        tokenizer_hash: str | None = None,
        max_length: int | None = None,
        hash: str | None = None,
        only_prompt: bool = False,
        template_name: Literal["qwen2.5-omni"] = "qwen2.5-omni",
        debug: bool = False,
        oss_time_log_thr: int = 10,
        add_eos_token: bool = True,
        add_bos_token: bool = False,
        use_audio_in_video: bool = True,
    ):
        # 从 processor_path 加载配置
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(processor_path, trust_remote_code=True)
        
        # 从 model_cfg 中提取需要的配置信息
        if hasattr(model_cfg, 'thinker_config'):
            vision_config = model_cfg.thinker_config.vision_config
            audio_config = getattr(model_cfg.thinker_config, 'audio_config', None)
        else:
            raise ValueError(
                f"Unexpected config structure. Expected thinker_config attribute. "
                f"Got type: {type(model_cfg)}"
            )

        self.oss_loader = None
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr
        if oss_loader_cfg is not None:
            self.oss_loader = Qwen25OmniOSSLoader(
                backend=oss_loader_cfg.backend,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
                **oss_loader_cfg.backend_kwargs,
            )

        self.only_prompt = only_prompt

        # 从 vision_config 读取配置
        self.image_size = getattr(vision_config, 'window_size', 448)
        self.patch_size = getattr(vision_config, 'patch_size', 14)
        
        if max_dynamic_patch is not None:
            max_num = max_dynamic_patch
        elif hasattr(model_cfg, 'max_dynamic_patch'):
            max_num = model_cfg.max_dynamic_patch
        elif hasattr(vision_config, 'max_dynamic_patch'):
            max_num = vision_config.max_dynamic_patch
        else:
            max_num = 6
        
        if min_dynamic_patch is not None:
            min_num = min_dynamic_patch
        elif hasattr(model_cfg, 'min_dynamic_patch'):
            min_num = model_cfg.min_dynamic_patch
        elif hasattr(vision_config, 'min_dynamic_patch'):
            min_num = vision_config.min_dynamic_patch
        else:
            min_num = 1
            
        self.max_dynamic_patch = max_num
        self.min_dynamic_patch = min_num
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames

        # Audio config
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.use_audio_in_video = use_audio_in_video
        
        self.dynamic_image_size = getattr(model_cfg, 'dynamic_image_size', True)
        self.use_thumbnail = getattr(model_cfg, 'use_thumbnail', True)
        self.data_name = os.path.basename(anno_name)
        self.data_augment = data_augment
        
        logger.info(
            f"[{self.data_name}] Using dynamic image size: {self.dynamic_image_size} and "
            f"max_dynamic_patch: {max_num} and min_dynamic_patch: {min_num} and "
            f"use_thumbnail: {self.use_thumbnail} data_aug: {self.data_augment} "
            f"audio_sample_rate: {self.audio_sample_rate} audio_max_length: {self.audio_max_length} for training."
        )
        
        spatial_merge_size = getattr(vision_config, 'spatial_merge_size', 2)
        self.downsample_ratio = 1.0 / spatial_merge_size
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 // (spatial_merge_size ** 2))
        
        if audio_config:
            self.num_audio_token = getattr(audio_config, 'max_source_positions', 1500)
        else:
            self.num_audio_token = 1500
        
        self.system_message = system_message

        self._hash_str = (
            f"{self.downsample_ratio}_{self.num_image_token}_{self.num_audio_token}_{self.system_message}_{self.use_thumbnail}"
            f"_{self.dynamic_image_size}_{self.max_num_frames}_{self.min_num_frames}"
            f"_{self.min_dynamic_patch}_{self.max_dynamic_patch}_{self.audio_sample_rate}_{self.audio_max_length}_{max_length}"
        )

        self.chat_template = CHAT_TEMPLATE_MAP[template_name]
        if system_message is not None:
            self.chat_template.default_system = system_message

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.image_context_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.video_context_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.chat_template.audio_context_token)
        
        # 必须要最后调用父类初始化
        super().__init__(tokenizer, self.chat_template, max_length, tokenizer_hash, hash)

    def _get_abs_path(self, path: str, media_root: str = "") -> str:
        """Get absolute path for media file."""
        if path.startswith(('http://', 'https://', 'oss://')):
            return path
        
        if os.path.isabs(path):
            return path
        
        if media_root:
            return os.path.join(media_root, path)
        
        return path

    def _get_transform(self):
        transform = build_transform(
            is_train=self.data_augment, input_size=self.image_size, pad2square=False, normalize_type="imagenet"
        )
        return transform

    def load_image(self, image_path: str):
        """Load and preprocess an image."""
        if self.oss_loader is not None and image_path.startswith("oss://"):
            start_time = time.time()
            image = self.oss_loader.load_image(image_path)
            load_time = time.time() - start_time
            if load_time > self.oss_time_log_thr:
                logger.warning(f"OSS image loading took {load_time:.2f}s for {image_path}")
        else:
            image = pil_loader(image_path)
        
        image = apply_exif_orientation(image)
        return image

    def load_video(self, video_path: str):
        """Load and preprocess a video with optional audio."""
        random_frame_num = generate_random_int_from_dict(
            {"video_path": video_path}, 
            self.min_num_frames, 
            self.max_num_frames
        )
        
        if self.oss_loader is not None and video_path.startswith("oss://"):
            start_time = time.time()
            video_data = self.oss_loader.load_video(
                video_path, 
                num_frames=random_frame_num,
                sample_rate=self.audio_sample_rate if self.use_audio_in_video else None
            )
            load_time = time.time() - start_time
            if load_time > self.oss_time_log_thr:
                logger.warning(f"OSS video loading took {load_time:.2f}s for {video_path}")
        else:
            video_data = read_qwen25_omni_video(
                video_path, 
                num_frames=random_frame_num,
                sample_rate=self.audio_sample_rate if self.use_audio_in_video else None
            )
        
        return video_data

    def load_audio(self, audio_path: str):
        """Load and preprocess an audio file."""
        if self.oss_loader is not None and audio_path.startswith("oss://"):
            start_time = time.time()
            audio = self.oss_loader.load_audio(
                audio_path,
                sample_rate=self.audio_sample_rate,
                max_length=self.audio_max_length
            )
            load_time = time.time() - start_time
            if load_time > self.oss_time_log_thr:
                logger.warning(f"OSS audio loading took {load_time:.2f}s for {audio_path}")
        else:
            audio = process_audio(
                audio_path,
                sample_rate=self.audio_sample_rate,
                max_length=self.audio_max_length
            )
        
        return audio

    def pure_text_get_item(self, data_item: dict) -> Qwen25OmniDataItem:
        messages = ChatMessages(messages=data_item["messages"])

        is_pretrain = False
        if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
            is_pretrain = True
        assert is_pretrain is False, "Text pretrain data should not be processed by this function"

        tokenized = messages.tokenize(self.tokenizer, self.chat_template)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
        ret = Qwen25OmniDataItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.randn(1, 3, self.image_size, self.image_size),
            audio_values=torch.randn(1, self.audio_max_length * self.audio_sample_rate),
            image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),  # 添加这个字段
            input_features=torch.randn(1, 128, 80),
            image_flags=torch.tensor([0] * 1, dtype=torch.long),
            audio_flags=torch.tensor([0] * 1, dtype=torch.long),
            num_tokens=len(input_ids),
            num_img_tokens=[0],
            num_audio_tokens=[0],
            num_imgs=[0],
            num_audios=[0],
            num_patches=[1],
        )
        return ret

    def calc_num_tokens_multi_modal_get_item(self, data_item: dict) -> CacheItem:
        num_image_tokens = []
        num_audio_tokens = []
        
        # Handle images
        if len(self._image_path) > 0:
            try:
                assert len(self._image_wh_list) >= 1, "image must have `hw` attribute when packing data"
                for size in self._image_wh_list:
                    if size[0] == 0 or size[1] == 0:
                        return {"num_tokens": 0}  # type: ignore
            except Exception as e:
                print(f"ERROR of image_wh: {e}, data_name: {self.data_name}")
                return {"num_tokens": 0}  # type: ignore

            num_tiles = []
            if self.dynamic_image_size:
                for size in self._image_wh_list:
                    num_patches = dynamic_num_patch(
                        size,
                        min_num=self.min_dynamic_patch,
                        max_num=max(1, self.max_dynamic_patch // len(self._image_path)),
                        image_size=self.image_size,
                        use_thumbnail=self.use_thumbnail,
                    )
                    num_tiles.append(num_patches)
            else:
                num_tiles = [1] * len(self._image_wh_list)

            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        # Handle audios
        if len(self._audio_path) > 0:
            num_audio_tokens = [self.num_audio_token] * len(self._audio_path)

        # Handle videos
        video_num_image_tokens = []
        video_has_audio = False
        if len(self._video_path) > 0:
            random_frame_num = generate_random_int_from_dict(data_item, self.min_num_frames, self.max_num_frames)
            n_frames = random_frame_num
            video_num_image_tokens = [self.num_image_token] * n_frames
            
            # Check if video has audio context
            for msg in data_item["messages"]:
                if "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        if "<AUDIO_IN_VIDEO_CONTEXT>" in content:
                            video_has_audio = True
                            break
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                if "<AUDIO_IN_VIDEO_CONTEXT>" in c.get("text", ""):
                                    video_has_audio = True
                                    break

        messages = ChatMessages(messages=data_item["messages"])

        try:
            if num_image_tokens:
                replace_image_token(messages, self.chat_template, num_image_tokens)
            if num_audio_tokens:
                replace_audio_token(messages, self.chat_template, num_audio_tokens)
            if video_num_image_tokens:
                replace_video_token(messages, self.chat_template, video_num_image_tokens)
            if video_has_audio:
                replace_audio_in_video_token(messages, self.chat_template, self.num_audio_token)
            
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)
            input_ids = tokenized["input_ids"]

            is_pretrain = False
            if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
                is_pretrain = True
            if is_pretrain:
                if self.add_bos_token:
                    input_ids = [self.bos_token_id] + input_ids
                if self.add_eos_token:
                    input_ids = input_ids + [self.eos_token_id]

            input_ids, _ = self._truncated_input_and_labels(input_ids)
            
            # Verify token counts
            input_ids_tensor = torch.tensor(input_ids)
            expected_img_tokens = sum(num_image_tokens) + sum(video_num_image_tokens)
            actual_img_tokens = (
                (input_ids_tensor == self.img_context_token_id).sum() + 
                (input_ids_tensor == self.video_context_token_id).sum()
            ).item()
            assert actual_img_tokens == expected_img_tokens, "ERROR: image/video tokens are truncated"
            
            expected_audio_tokens = sum(num_audio_tokens) + (self.num_audio_token if video_has_audio else 0)
            actual_audio_tokens = (input_ids_tensor == self.audio_context_token_id).sum().item()
            assert actual_audio_tokens == expected_audio_tokens, "ERROR: audio tokens are truncated"
            
            return {"num_tokens": len(input_ids)}
        except Exception as e:
            print(f"ERROR of Preprocess function: {e}, data_name: {self.data_name}")
            return {"num_tokens": 0}

    def multi_modal_get_item(self, item: dict, media_root: str = "") -> Qwen25OmniDataItem | None:
        """Process multimodal data including images, videos, and audio."""
        messages = ChatMessages(messages=item["messages"])
        
        # 加载图像
        images = []
        image_sizes = []
        for image_path_ in self._image_path:
            image_path = self._get_abs_path(image_path_, media_root)
            try:
                image = self.load_image(image_path)
                images.append(image)
                image_sizes.append(image.size)
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                if self.debug:
                    raise
                return None
        
        # 加载视频
        videos = []
        video_audios = []
        for video_path_ in self._video_path:
            video_path = self._get_abs_path(video_path_, media_root)
            try:
                video_data = self.load_video(video_path)
                videos.append(video_data["video"])
                
                if self.use_audio_in_video and video_data.get("audio") is not None:
                    video_audios.append(video_data["audio"])
            except Exception as e:
                logger.error(f"Error loading video {video_path}: {e}")
                if self.debug:
                    raise
                return None
        
        # 加载纯音频文件
        audios = []
        for audio_path_ in self._audio_path:
            audio_path = self._get_abs_path(audio_path_, media_root)
            try:
                audio = self.load_audio(audio_path)
                audios.append(audio)
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                if self.debug:
                    raise
                return None
        
        # 处理图像
        processed_images = []
        num_image_tokens = []
        image_grid_thws = []
        
        transform = self._get_transform()
        
        for image, size in zip(images, image_sizes):
            if self.dynamic_image_size:
                patches = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                
                pixel_values = [transform(patch) for patch in patches]
                pixel_values = torch.stack(pixel_values)
                
                num_patches = len(patches)
                num_image_tokens.append(self.num_image_token * num_patches)
                
                grid_t = 1
                grid_h = grid_w = int(np.sqrt(num_patches))
                image_grid_thws.append([grid_t, grid_h, grid_w])
                
            else:
                pixel_values = transform(image).unsqueeze(0)
                num_image_tokens.append(self.num_image_token)
                image_grid_thws.append([1, 1, 1])
            
            processed_images.append(pixel_values)
        
        # 处理视频帧
        processed_videos = []
        num_video_tokens = []
        video_grid_thws = []
        
        for video in videos:
            num_frames = len(video)
            video_patches = []
            
            for frame in video:
                from PIL import Image
                frame_pil = Image.fromarray(frame)
                frame_tensor = transform(frame_pil)
                video_patches.append(frame_tensor)
            
            video_tensor = torch.stack(video_patches)
            processed_videos.append(video_tensor)
            
            num_video_tokens.extend([self.num_image_token] * num_frames)
            video_grid_thws.extend([[1, 1, 1]] * num_frames)
        
        # 处理音频
        processed_audios = []
        num_audio_tokens_list = []
        
        for audio in video_audios:
            processed_audios.append(torch.from_numpy(audio).float())
            num_audio_tokens_list.append(self.num_audio_token)
        
        for audio in audios:
            processed_audios.append(torch.from_numpy(audio).float())
            num_audio_tokens_list.append(self.num_audio_token)
        
        # 替换token占位符
        try:
            if num_image_tokens:
                replace_image_token(messages, self.chat_template, num_image_tokens)
            if num_audio_tokens_list:
                replace_audio_token(messages, self.chat_template, num_audio_tokens_list)
            if num_video_tokens:
                replace_video_token(messages, self.chat_template, num_video_tokens)
            if video_audios:
                replace_audio_in_video_token(messages, self.chat_template, self.num_audio_token)
            
            tokenized = messages.tokenize(self.tokenizer, self.chat_template)
            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            
            is_pretrain = False
            if len(messages.messages) == 1 and messages.messages[0].role == "pretrain":
                is_pretrain = True
            
            if is_pretrain:
                if self.add_bos_token:
                    input_ids = [self.bos_token_id] + input_ids
                    labels = [self.bos_token_id] + labels
                if self.add_eos_token:
                    input_ids = input_ids + [self.eos_token_id]
                    labels = labels + [self.eos_token_id]
            
            input_ids, labels = self._truncated_input_and_labels(input_ids, labels)
            
            # 合并所有pixel_values
            if processed_images or processed_videos:
                all_pixel_values = processed_images + processed_videos
                pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else torch.randn(1, 3, self.image_size, self.image_size)
            else:
                pixel_values = torch.randn(1, 3, self.image_size, self.image_size)
            
            # 合并image_grid_thw
            if image_grid_thws or video_grid_thws:
                all_grid_thws = image_grid_thws + video_grid_thws
                image_grid_thw = torch.tensor(all_grid_thws, dtype=torch.long)
            else:
                image_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.long)
            
            # 合并所有audio_values
            if processed_audios:
                max_audio_len = max(a.shape[0] for a in processed_audios)
                padded_audios = []
                for audio in processed_audios:
                    if audio.shape[0] < max_audio_len:
                        padding = torch.zeros(max_audio_len - audio.shape[0])
                        audio = torch.cat([audio, padding])
                    padded_audios.append(audio)
                audio_values = torch.stack(padded_audios)
            else:
                audio_values = torch.randn(1, self.audio_max_length * self.audio_sample_rate)
            
            # 构建返回对象
            ret = Qwen25OmniDataItem(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,  # 确保包含这个字段
                audio_values=audio_values,
                image_flags=torch.tensor([1 if (images or videos) else 0] * len(pixel_values), dtype=torch.long),
                audio_flags=torch.tensor([1 if processed_audios else 0] * len(audio_values), dtype=torch.long),
                num_tokens=len(input_ids),
                num_img_tokens=num_image_tokens + num_video_tokens,
                num_audio_tokens=num_audio_tokens_list,
                num_imgs=[len(images) + sum(len(v) for v in videos)],
                num_audios=[len(processed_audios)],
                num_patches=[len(pixel_values)],
            )
            
            return ret
            
        except Exception as e:
            logger.error(f"Error in multi_modal_get_item: {e}")
            if self.debug:
                raise
            return None

    def __call__(self, item: dict, media_root: str = "") -> Qwen25OmniDataItem | None:
        """Process an item with multimodal data."""
        # 初始化路径列表
        self._image_path = []
        self._video_path = []
        self._audio_path = []
        self._image_wh_list = []
        
        # 收集所有模态的路径
        for message in item.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, str):
                continue
                
            for c in content:
                if not isinstance(c, dict):
                    continue
                    
                # 处理图像
                if c.get("type") == "image_url":
                    image_url = c.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    if url:
                        self._image_path.append(url)
                        width = c.get("width", 0)
                        height = c.get("height", 0)
                        self._image_wh_list.append((width, height))
                
                # 处理视频
                elif c.get("type") == "video":
                    video_url = c.get("video", "")
                    if video_url:
                        self._video_path.append(video_url)
                
                # 处理音频
                elif c.get("type") == "audio":
                    audio_url = c.get("audio", "")
                    if audio_url:
                        self._audio_path.append(audio_url)
        
        # 调用对应的处理逻辑
        if self.only_prompt:
            return self.prompt_only_get_item(item, media_root)
        elif self._image_path or self._video_path or self._audio_path:
            return self.multi_modal_get_item(item, media_root)
        else:
            return self.pure_text_get_item(item)


class Qwen25OmniTokenizeFnConfig(BaseMLLMTokenizeFnConfig):
    model_config = ConfigDict(title="Qwen2.5-Omni tokenize function config", extra="forbid")
    
    processor_path: str
    
    max_dynamic_patch: int | None = None
    min_dynamic_patch: int | None = None
    min_num_frames: int = 4
    max_num_frames: int = 24
    audio_sample_rate: int = 16000
    audio_max_length: int = 30
    data_augment: bool = False
    oss_loader_cfg: OSSLoaderConfig | None = None
    template_name: Literal["qwen2.5-omni"] = "qwen2.5-omni"
    use_audio_in_video: bool = True

    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> Qwen25OmniTokenizeFunction:
        return Qwen25OmniTokenizeFunction(
            tokenizer,
            self.processor_path,
            anno_name,
            max_length=self.max_length,
            tokenizer_hash=tokenizer_hash,
            max_dynamic_patch=self.max_dynamic_patch,
            min_dynamic_patch=self.min_dynamic_patch,
            data_augment=self.data_augment,
            system_message=self.system_message,
            min_num_frames=self.min_num_frames,
            max_num_frames=self.max_num_frames,
            audio_sample_rate=self.audio_sample_rate,
            audio_max_length=self.audio_max_length,
            oss_loader_cfg=self.oss_loader_cfg,
            template_name=self.template_name,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            add_eos_token=self.add_eos_token,
            add_bos_token=self.add_bos_token,
            use_audio_in_video=self.use_audio_in_video, 
        )
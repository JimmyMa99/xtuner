# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
from typing import Literal, Union
from pathlib import Path

import torch
import torchaudio
import numpy as np

# Import all utilities from intern_s1_vl_utils
from .intern_s1_vl_utils import (
    InternS1VLOSSLoader,
    pil_loader,
    extract_frame_number,
    sort_frames,
    get_frame_indices,
    read_frames_folder,
    read_frames_gif,
    read_frames_decord,
    read_interns1_vl_video,
)

def pil_loader(path: str):
    """Load an image from path using PIL."""
    from PIL import Image
    
    # 直接打开文件
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def read_qwen25_omni_video(
    video_path: Union[str, Path],
    num_frames: int = 8,
    fps: int = 1,
) -> torch.Tensor:
    """
    Read video file and extract frames for Qwen2.5-Omni.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        fps: Frames per second to sample
    
    Returns:
        Video tensor of shape [num_frames, channels, height, width]
    """
    video_path = str(video_path)
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video from {video_path}: {e}")
    
    total_frames = len(vr)
    
    # Calculate frame indices to extract
    if total_frames <= num_frames:
        # If video has fewer frames than requested, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    # Extract frames
    frames = vr.get_batch(frame_indices).asnumpy()  # [num_frames, H, W, C]
    
    # Convert to tensor and rearrange dimensions
    # From [num_frames, H, W, C] to [num_frames, C, H, W]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    
    # Normalize to [0, 1] if needed
    if frames.max() > 1.0:
        frames = frames / 255.0
    
    return frames

def read_audio_file(audio_path, sample_rate=16000, max_duration=30.0, client=None):
    """Read audio file and return waveform.
    
    Args:
        audio_path: path to audio file (local or s3://)
        sample_rate: target sample rate
        max_duration: maximum duration in seconds
        client: OSS client for s3 paths
        
    Returns:
        waveform tensor of shape [num_samples]
    """
    if "s3://" in audio_path:
        assert client is not None, "client should be provided for s3 backend"
        audio_bytes = client.get(audio_path)
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    else:
        waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    
    # Truncate or pad to max_duration
    max_samples = int(max_duration * sample_rate)
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    elif waveform.shape[0] < max_samples:
        padding = max_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding), value=0)
    
    return waveform


class Qwen25OmniOSSLoader(InternS1VLOSSLoader):
    """Extended OSS Loader with audio support for Qwen2.5-Omni."""
    
    def __call__(
        self,
        path,
        image_type="image",
        max_num_frames=-1,
        min_num_frames=8,
        sample="rand",
        clip=None,
        random_frame_num=None,
        # Audio specific parameters
        audio_sample_rate=16000,
        audio_max_duration=30.0,
    ):
        if image_type == "audio":
            return read_audio_file(
                path,
                sample_rate=audio_sample_rate,
                max_duration=audio_max_duration,
                client=self.client,
            )
        else:
            # Delegate to parent class for image/video
            return super().__call__(
                path=path,
                image_type=image_type,
                max_num_frames=max_num_frames,
                min_num_frames=min_num_frames,
                sample=sample,
                clip=clip,
                random_frame_num=random_frame_num,
            )


__all__ = [
    'InternS1VLOSSLoader',
    'Qwen25OmniOSSLoader',
    'pil_loader',
    'extract_frame_number',
    'sort_frames',
    'get_frame_indices',
    'read_frames_folder',
    'read_frames_gif',
    'read_frames_decord',
    'read_interns1_vl_video',
    'read_audio_file',
]
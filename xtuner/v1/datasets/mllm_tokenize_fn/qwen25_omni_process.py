# Copyright (c) OpenMMLab. All rights reserved.

import math
import warnings
from typing import Tuple

import numpy as np
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    """Build image transform pipeline.
    
    Args:
        is_train: whether in training mode
        input_size: target image size
        pad2square: whether to pad image to square
        normalize_type: normalization type, 'imagenet' or other
    """
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios.
    
    Args:
        aspect_ratio: original aspect ratio
        target_ratios: list of target aspect ratios
        width: image width
        height: image height
        image_size: target image size
    
    Returns:
        best_ratio: the closest aspect ratio
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into multiple patches.
    
    Args:
        image: PIL Image
        min_num: minimum number of patches
        max_num: maximum number of patches
        image_size: target image size for each patch
        use_thumbnail: whether to add a thumbnail patch
    
    Returns:
        list of PIL Images (patches)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def dynamic_num_patch(image_size, min_num=1, max_num=12, image_size_target=448, use_thumbnail=False):
    """Calculate the number of patches for dynamic preprocessing.
    
    Args:
        image_size: tuple of (width, height)
        min_num: minimum number of patches
        max_num: maximum number of patches
        image_size_target: target image size for each patch
        use_thumbnail: whether to add a thumbnail patch
    
    Returns:
        number of patches
    """
    orig_width, orig_height = image_size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size_target)

    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    if use_thumbnail and blocks != 1:
        blocks += 1
    
    return blocks


def process_audio(audio_data, sample_rate=16000, max_length=30):
    """Process audio data.
    
    Args:
        audio_data: audio waveform tensor or numpy array
        sample_rate: target sample rate
        max_length: maximum audio length in seconds
    
    Returns:
        processed audio tensor of shape [max_length * sample_rate]
    """
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data).float()
    
    # Ensure audio is 1D
    if audio_data.dim() > 1:
        audio_data = audio_data.mean(dim=0)
    
    # Resample if needed (this is a simplified version, you may need torchaudio.transforms.Resample)
    target_length = max_length * sample_rate
    current_length = audio_data.shape[0]
    
    if current_length > target_length:
        # Truncate
        audio_data = audio_data[:target_length]
    elif current_length < target_length:
        # Pad
        padding = target_length - current_length
        audio_data = torch.nn.functional.pad(audio_data, (0, padding), mode='constant', value=0)
    
    return audio_data


def extract_audio_from_video(video_path, sample_rate=16000):
    """Extract audio from video file.
    
    Args:
        video_path: path to video file
        sample_rate: target sample rate
    
    Returns:
        audio waveform tensor
    """
    try:
        import subprocess
        import tempfile
        import os
        
        # Create a temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
        
        # Extract audio using ffmpeg
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # no video
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',  # mono
            '-y',  # overwrite
            tmp_audio_path
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Load audio
        waveform, sr = torchaudio.load(tmp_audio_path)
        
        # Clean up
        os.remove(tmp_audio_path)
        
        # Ensure correct sample rate
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    except Exception as e:
        warnings.warn(f"Failed to extract audio from video {video_path}: {e}")
        # Return silence
        return torch.zeros(sample_rate * 30)


def load_video_frames(video_path, num_frames=8, sample='rand', clip=None):
    """Load video frames using decord.
    
    Args:
        video_path: path to video file
        num_frames: number of frames to sample
        sample: sampling strategy, 'rand' or 'uniform'
        clip: optional clip range (start_time, end_time) in seconds
    
    Returns:
        list of PIL Images
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        
        # Handle clip
        if clip is not None:
            start_time, end_time = clip
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            start_frame = max(0, start_frame)
            end_frame = min(total_frame_num, end_frame)
        else:
            start_frame = 0
            end_frame = total_frame_num
        
        # Sample frames
        frame_idx = []
        if sample == 'rand':
            # Random sampling
            available_frames = end_frame - start_frame
            if available_frames <= num_frames:
                frame_idx = list(range(start_frame, end_frame))
            else:
                frame_idx = sorted(np.random.choice(
                    range(start_frame, end_frame), 
                    size=num_frames, 
                    replace=False
                ).tolist())
        else:  # uniform sampling
            frame_idx = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()
        
        # Load frames
        frames = vr.get_batch(frame_idx).asnumpy()
        images = [Image.fromarray(frame) for frame in frames]
        
        return images
    
    except Exception as e:
        warnings.warn(f"Failed to load video {video_path}: {e}")
        # Return dummy images
        return [Image.new('RGB', (224, 224), color='black') for _ in range(num_frames)]


def resample_audio(waveform, orig_sr, target_sr):
    """Resample audio to target sample rate.
    
    Args:
        waveform: audio waveform tensor
        orig_sr: original sample rate
        target_sr: target sample rate
    
    Returns:
        resampled waveform
    """
    if orig_sr == target_sr:
        return waveform
    
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def pad_or_truncate_audio(waveform, target_length):
    """Pad or truncate audio to target length.
    
    Args:
        waveform: audio waveform tensor of shape [channels, samples] or [samples]
        target_length: target number of samples
    
    Returns:
        processed waveform of shape [channels, target_length] or [target_length]
    """
    if waveform.dim() == 1:
        current_length = waveform.shape[0]
        if current_length > target_length:
            return waveform[:target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        return waveform
    else:  # 2D tensor
        current_length = waveform.shape[1]
        if current_length > target_length:
            return waveform[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        return waveform


def compute_mel_spectrogram(waveform, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
    """Compute mel spectrogram from waveform.
    
    Args:
        waveform: audio waveform tensor
        sample_rate: sample rate
        n_fft: FFT size
        hop_length: hop length
        n_mels: number of mel bins
    
    Returns:
        mel spectrogram tensor
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    
    return mel_spec


def normalize_audio(waveform):
    """Normalize audio waveform to [-1, 1].
    
    Args:
        waveform: audio waveform tensor
    
    Returns:
        normalized waveform
    """
    max_val = torch.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


class AudioTransform:
    """Audio transform class for preprocessing audio data."""
    
    def __init__(self, sample_rate=16000, max_length=30, normalize=True, to_mel=False, n_mels=80):
        """
        Args:
            sample_rate: target sample rate
            max_length: maximum audio length in seconds
            normalize: whether to normalize audio
            to_mel: whether to convert to mel spectrogram
            n_mels: number of mel bins if to_mel is True
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.target_length = max_length * sample_rate
        self.normalize = normalize
        self.to_mel = to_mel
        self.n_mels = n_mels
    
    def __call__(self, waveform, orig_sr=None):
        """
        Args:
            waveform: audio waveform tensor or numpy array
            orig_sr: original sample rate (if provided, will resample)
        
        Returns:
            processed audio tensor
        """
        # Convert to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        # Resample if needed
        if orig_sr is not None and orig_sr != self.sample_rate:
            waveform = resample_audio(waveform, orig_sr, self.sample_rate)
        
        # Convert to mono if needed
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        
        # Normalize
        if self.normalize:
            waveform = normalize_audio(waveform)
        
        # Pad or truncate
        waveform = pad_or_truncate_audio(waveform, self.target_length)
        
        # Convert to mel spectrogram if needed
        if self.to_mel:
            waveform = compute_mel_spectrogram(waveform, self.sample_rate, n_mels=self.n_mels)
        
        return waveform


def build_audio_transform(sample_rate=16000, max_length=30, normalize=True, to_mel=False, n_mels=80):
    """Build audio transform.
    
    Args:
        sample_rate: target sample rate
        max_length: maximum audio length in seconds
        normalize: whether to normalize audio
        to_mel: whether to convert to mel spectrogram
        n_mels: number of mel bins
    
    Returns:
        AudioTransform instance
    """
    return AudioTransform(
        sample_rate=sample_rate,
        max_length=max_length,
        normalize=normalize,
        to_mel=to_mel,
        n_mels=n_mels
    )


def split_to_patches(image, patch_size):
    """Split image into patches.
    
    Args:
        image: PIL Image or tensor
        patch_size: size of each patch
    
    Returns:
        list of patches
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        patches = []
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                box = (j, i, min(j + patch_size, width), min(i + patch_size, height))
                patch = image.crop(box)
                # Resize if needed to ensure all patches are the same size
                if patch.size != (patch_size, patch_size):
                    patch = patch.resize((patch_size, patch_size))
                patches.append(patch)
        return patches
    else:
        raise NotImplementedError("Tensor input not implemented yet")


def merge_patches(patches, grid_size):
    """Merge patches back into an image.
    
    Args:
        patches: list of PIL Images
        grid_size: tuple of (rows, cols)
    
    Returns:
        merged PIL Image
    """
    rows, cols = grid_size
    patch_size = patches[0].size[0]
    
    merged_image = Image.new('RGB', (cols * patch_size, rows * patch_size))
    
    for idx, patch in enumerate(patches):
        i = idx // cols
        j = idx % cols
        merged_image.paste(patch, (j * patch_size, i * patch_size))
    
    return merged_image
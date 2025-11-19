import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from typing import Optional

from xtuner.v1.utils import get_device

DEVICE = get_device()


def init_world_mesh():
    """
    Initialize device mesh for FSDP.
    
    Returns:
        DeviceMesh for FSDP operations
    """
    if not dist.is_initialized():
        return None
    
    device = DEVICE
    world_size = dist.get_world_size()
    
    # Create a 1D mesh for FSDP
    # TODO: Support HSDP (Hybrid Sharded Data Parallel) with 2D mesh
    fsdp_mesh = init_device_mesh(device, (world_size,))
    
    return fsdp_mesh


def get_chunked_index(
    token_indices: torch.Tensor, 
    tokens_per_chunk: int, 
    remove_index: int = 0
) -> list[tuple[int, int]]:
    """
    Splits token index list into chunks based on token value ranges.
    
    Given a list of token indices, returns a list of (start, end) index tuples representing
    slices of the list where the token values fall within successive ranges of `tokens_per_chunk`.
    
    For example, if `tokens_per_chunk` is 1000, the function will create chunks such that:
    - the first chunk contains token values < 1000,
    - the second chunk contains values >= 1000 and < 2000, and so on.
    
    Args:
        token_indices: A monotonically increasing list/tensor of token index values
        tokens_per_chunk: Number of tokens per chunk (used as the chunk size threshold)
        remove_index: An index id to subtract from `token_indices` before chunking
    
    Returns:
        A list of tuples, each representing the start (inclusive) and end (exclusive) 
        indices of a chunk in `token_indices`
    """
    if isinstance(token_indices, torch.Tensor):
        token_indices = token_indices.cpu().numpy()
    
    def _iter():
        i, start_idx = 0, 0
        current_chunk = 1
        while i < len(token_indices):
            if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                yield (start_idx, i)
                start_idx = i
                current_chunk += 1
            i += 1
        yield (start_idx, len(token_indices))
    
    return list(_iter())


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: list[int],
    grid_ws: list[int],
) -> torch.Tensor:
    """
    Calculate position IDs for vision tokens in the language model.
    
    Args:
        start_idx: Starting position index
        vision_idx: Index of current vision input
        spatial_merge_size: Spatial merge size for vision features
        t_index: Temporal indices
        grid_hs: Grid heights for each vision input
        grid_ws: Grid widths for each vision input
    
    Returns:
        Position IDs tensor of shape [3, num_tokens] for temporal, height, width
    """
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index_tensor = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    
    _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    
    return llm_pos_ids


def calculate_rope_indices(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_seqlens: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    spatial_merge_size: int = 2,
    position_id_per_seconds: int = 25,
    seconds_per_chunk: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    audio_token_id: int = 151646,
    vision_start_token_id: int = 151652,
    audio_start_token_id: int = 151647,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate 3D RoPE indices for multimodal inputs.
    
    This function computes position IDs for the multimodal rotary position embeddings (M-RoPE)
    used in Qwen2.5-Omni models. It handles temporal, height, and width dimensions for 
    vision inputs, and temporal dimension for audio inputs.
    
    Args:
        input_ids: Input token IDs
        image_grid_thw: Grid dimensions (temporal, height, width) for images
        video_grid_thw: Grid dimensions (temporal, height, width) for videos
        audio_seqlens: Sequence lengths for audio inputs
        attention_mask: Attention mask
        use_audio_in_video: Whether to use audio track in video
        spatial_merge_size: Spatial merge size for vision features
        position_id_per_seconds: Position ID increment per second
        seconds_per_chunk: Duration in seconds of each chunk
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders
        audio_token_id: Token ID for audio placeholders
        vision_start_token_id: Token ID for vision start marker
        audio_start_token_id: Token ID for audio start marker
        second_per_grids: Seconds per grid for each video
    
    Returns:
        Tuple of (position_ids, mrope_position_deltas)
        - position_ids: shape [3, batch_size, sequence_length]
        - mrope_position_deltas: shape [batch_size, 1]
    """
    mrope_position_deltas = []
    
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None):
        total_input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask == 1
        
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i]]
            
            # Count different modalities
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1] if len(vision_start_indices) > 0 else torch.tensor([])
            
            audio_nums = torch.sum(input_ids == audio_start_token_id).item()
            image_nums = (vision_tokens == image_token_id).sum().item() if len(vision_tokens) > 0 else 0
            video_nums = (
                (vision_tokens == audio_start_token_id).sum().item()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum().item()
            ) if len(vision_tokens) > 0 else 0
            
            input_tokens = input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            
            image_idx, video_idx, audio_idx = 0, 0, 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            
            multimodal_nums = (
                image_nums + audio_nums if use_audio_in_video 
                else image_nums + video_nums + audio_nums
            )
            
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                
                # Find next modality token
                ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens[st:] and remain_images > 0 else len(input_tokens) + 1
                ed_video = input_tokens.index(video_token_id, st) if video_token_id in input_tokens[st:] and remain_videos > 0 else len(input_tokens) + 1
                ed_audio = input_tokens.index(audio_token_id, st) if audio_token_id in input_tokens[st:] and remain_audios > 0 else len(input_tokens) + 1
                
                min_ed = min(ed_image, ed_video, ed_audio)
                
                if min_ed == ed_audio:
                    # Process audio
                    text_len = min_ed - st - 1
                    if text_len > 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    
                    # BOS token
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                    
                    # Audio features
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids_list.append(torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx)
                    
                    # EOS token
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                    
                    st += text_len + 1 + audio_len + 1
                    audio_idx += 1
                    remain_audios -= 1
                
                elif min_ed == ed_image:
                    # Process image
                    text_len = min_ed - st - 1
                    if text_len > 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    
                    # BOS token
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                    
                    # Image features with 3D positions
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size ** 2)
                    llm_pos_ids_list.append(llm_pos_ids)
                    
                    # EOS token
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                    
                    st += text_len + 1 + image_len + 1
                    image_idx += 1
                    remain_images -= 1
                
                elif min_ed == ed_video:
                    if not use_audio_in_video:
                        # Process video without audio
                        text_len = min_ed - st - 1
                        if text_len > 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        
                        # BOS token
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        
                        # Video features
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        
                        if second_per_grids is not None:
                            t_index = (
                                torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                            ).long()
                        else:
                            t_index = (torch.arange(grid_t) * position_id_per_seconds).long()
                        
                        llm_pos_ids = get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                        llm_pos_ids_list.append(llm_pos_ids)
                        
                        # EOS token
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        
                        st += text_len + 1 + video_len + 1
                        video_idx += 1
                        remain_videos -= 1
                    else:
                        # Process video with audio (interleaved)
                        text_len = min_ed - st - 2
                        if text_len > 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        
                        # BOS tokens (vision and audio)
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        
                        # Calculate position IDs for audio and video
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        
                        if second_per_grids is not None:
                            t_index = (
                                torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                            ).long()
                        else:
                            t_index = (torch.arange(grid_t) * position_id_per_seconds).long()
                        
                        video_llm_pos_ids = get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        
                        # Interleave video and audio chunks
                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_chunk_index = video_chunk_indexes[j]
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0]:video_chunk_index[1]]
                                )
                            if j < len(audio_chunk_indexes):
                                audio_chunk_index = audio_chunk_indexes[j]
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0]:audio_chunk_index[1]]
                                )
                        
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size ** 2)
                        
                        # EOS tokens
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(1).view(1, -1).expand(3, -1) + st_idx)
                        
                        st += text_len + 2 + audio_len + video_len + 2
                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
            
            # Remaining text tokens
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            
            if attention_mask is not None:
                position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
            else:
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
            
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        
        mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
        
        return position_ids, mrope_position_deltas
    else:
        # Fallback for text-only inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        
        return position_ids, mrope_position_deltas
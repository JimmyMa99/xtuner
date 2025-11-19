import os
from unittest import TestCase
from xtuner.v1.datasets import Qwen25OmniTokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor
import json
import torch
import parametrize

QWEN2_5_OMNI_PATH = os.environ.get("QWEN2_5_OMNI_PATH", "/root/models/real_qwenomni/Qwen/Qwen2.5-Omni-3B")
VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "tests/")
AUDIO_ROOT = os.environ.get("AUDIO_ROOT", "tests/")


class TestQwen25OmniTokenizeFn(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN2_5_OMNI_PATH)
        self.tokenize_fn = Qwen25OmniTokenizeFnConfig(processor_path=QWEN2_5_OMNI_PATH).build(self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(QWEN2_5_OMNI_PATH)

    def test_qwen2_5_omni_single_image(self):
        """Test single image processing"""
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                del messages[0]['content'][0]['image_url']

                # Remove <IMG_CONTEXT> to align with HF format
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>', '<|vision_bos|><|IMAGE|><|vision_eos|>')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    def test_qwen2_5_omni_multi_image(self):
        """Test multiple images processing"""
        data_path = 'tests/resource/mllm_sft_multi_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                messages[0]['content'][1]['type'] = 'image'
                messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image_url']['url']
                del messages[0]['content'][0]['image_url']
                del messages[0]['content'][1]['image_url']
                
                # Replace placeholders
                text = messages[0]['content'][2]['text']
                text = text.replace('<IMG_CONTEXT>', '<|vision_bos|><|IMAGE|><|vision_eos|>')
                text = text.replace('\n', '')
                messages[0]['content'][2]['text'] = text
                
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    def test_qwen2_5_omni_single_video(self):
        """Test single video processing without audio"""
        data_path = 'tests/resource/mllm_sft_single_video_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                video_grid_thw_xtuner: torch.Tensor = ret['video_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'video'
                messages[0]['content'][0]['video'] = ['tests/' + messages[0]['content'][0]['video_url']['url']]
                del messages[0]['content'][0]['video_url']

                # Replace <VIDEO_CONTEXT> with HF format
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace(
                    '<VIDEO_CONTEXT>', '<|vision_bos|><|VIDEO|><|vision_eos|>')
                
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                video_grid_thw_hf = ret['video_grid_thw']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(video_grid_thw_xtuner, video_grid_thw_hf))

    @parametrize.parametrize("use_audio_in_video", [(True,), (False,)])
    def test_qwen2_5_omni_video_with_audio(self, use_audio_in_video):
        """Test video processing with audio track"""
        tokenize_fn = Qwen25OmniTokenizeFnConfig(
            processor_path=QWEN2_5_OMNI_PATH,
            use_audio_in_video=use_audio_in_video
        ).build(self.tokenizer)
        
        data_path = 'tests/resource/mllm_sft_video_with_audio_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                video_grid_thw_xtuner: torch.Tensor = ret['video_grid_thw']
                
                if use_audio_in_video:
                    input_features_xtuner = ret['input_features']
                    feature_attention_mask_xtuner = ret['feature_attention_mask']
                    video_second_per_grid_xtuner = ret['video_second_per_grid']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'video'
                messages[0]['content'][0]['video'] = ['tests/' + messages[0]['content'][0]['video_url']['url']]
                del messages[0]['content'][0]['video_url']

                # Replace placeholder based on use_audio_in_video
                if use_audio_in_video:
                    messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace(
                        '<AUDIO_IN_VIDEO_CONTEXT>', '<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>')
                else:
                    messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace(
                        '<VIDEO_CONTEXT>', '<|vision_bos|><|VIDEO|><|vision_eos|>')
                
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=False, 
                    tokenize=True,
                    return_dict=True
                )
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                video_grid_thw_hf = ret['video_grid_thw']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(video_grid_thw_xtuner, video_grid_thw_hf))
                
                if use_audio_in_video:
                    input_features_hf = ret['input_features']
                    feature_attention_mask_hf = ret['feature_attention_mask']
                    video_second_per_grid_hf = ret['video_second_per_grid']
                    
                    self.assertTrue(torch.allclose(input_features_xtuner, input_features_hf))
                    self.assertTrue(torch.allclose(feature_attention_mask_xtuner, feature_attention_mask_hf))
                    self.assertTrue(torch.allclose(video_second_per_grid_xtuner, video_second_per_grid_hf))

    def test_qwen2_5_omni_pure_audio(self):
        """Test pure audio processing"""
        data_path = 'tests/resource/mllm_sft_audio_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                input_features_xtuner = ret['input_features']
                feature_attention_mask_xtuner = ret['feature_attention_mask']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'audio'
                messages[0]['content'][0]['audio'] = ['tests/' + messages[0]['content'][0]['audio_url']['url']]
                del messages[0]['content'][0]['audio_url']

                # Replace <AUDIO_CONTEXT> with HF format
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace(
                    '<AUDIO_CONTEXT>', '<|audio_bos|><|AUDIO|><|audio_eos|>')
                
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                input_features_hf = ret['input_features']
                feature_attention_mask_hf = ret['feature_attention_mask']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(input_features_xtuner, input_features_hf))
                self.assertTrue(torch.allclose(feature_attention_mask_xtuner, feature_attention_mask_hf))

    def test_qwen2_5_omni_pure_text(self):
        """Test pure text processing"""
        data_path = 'tests/resource/mllm_sft_text_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data)
                input_ids_xtuner = ret['input_ids']

                # to hf openai format
                messages = raw_data['messages']
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                self.assertEqual(input_ids_xtuner, input_ids_hf)

    def test_qwen2_5_omni_mixed_modalities(self):
        """Test mixed modalities: image + audio + text"""
        data_path = 'tests/resource/mllm_sft_mixed_modality_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner = ret['pixel_values']
                image_grid_thw_xtuner = ret['image_grid_thw']
                input_features_xtuner = ret['input_features']
                feature_attention_mask_xtuner = ret['feature_attention_mask']

                # to hf openai format
                messages = raw_data['messages']
                # Assuming first is image, second is audio
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                del messages[0]['content'][0]['image_url']
                
                messages[0]['content'][1]['type'] = 'audio'
                messages[0]['content'][1]['audio'] = ['tests/' + messages[0]['content'][1]['audio_url']['url']]
                del messages[0]['content'][1]['audio_url']

                # Replace placeholders
                text = messages[0]['content'][2]['text']
                text = text.replace('<IMG_CONTEXT>', '<|vision_bos|><|IMAGE|><|vision_eos|>')
                text = text.replace('<AUDIO_CONTEXT>', '<|audio_bos|><|AUDIO|><|audio_eos|>')
                messages[0]['content'][2]['text'] = text
                
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                input_features_hf = ret['input_features']
                feature_attention_mask_hf = ret['feature_attention_mask']
                
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))
                self.assertTrue(torch.allclose(input_features_xtuner, input_features_hf))
                self.assertTrue(torch.allclose(feature_attention_mask_xtuner, feature_attention_mask_hf))


if __name__ == '__main__':
    import unittest
    
    # 设置环境变量（如果还没设置）
    if "QWEN2_5_OMNI_PATH" not in os.environ:
        print("Warning: QWEN2_5_OMNI_PATH not set, using default path")
        os.environ["QWEN2_5_OMNI_PATH"] = "/root/models/real_qwenomni/Qwen/Qwen2.5-Omni-3B"
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQwen25OmniTokenizeFn)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("="*70)
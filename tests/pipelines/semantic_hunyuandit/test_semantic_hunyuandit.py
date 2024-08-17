import gc
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    HunyuanDiT2DModel,
    SemanticHunyuanDiTPipeline,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class SemanticHunyuanDiTPipelineTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SemanticHunyuanDiTPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HunyuanDiT2DModel(
            sample_size=16,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            in_channels=4,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            hidden_size=24,
            activation_fn="gelu-approximate",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDPMScheduler()
        text_encoder = BertModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "edit_prompt": "fries",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "use_resolution_binning": False,
        }
        return inputs

    def test_semantic_hunyuan_pipeline(self):
        pass  # todo

    def test_semantic_hunyuan_pipeline_with_edit(self):
        pass  # todo

    def test_inference_batch_single_identical(self):
        pass  # todo

    def test_attention_slicing_forward_pass(self):
        pass  # todo

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_semantic_hunyuan_pipeline_fp16(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch.float16)

        inputs = self.get_dummy_inputs(torch_device)
        image = pipe(**inputs).images

        self.assertEqual(image.shape, (1, 16, 16, 3))

    @slow
    @require_torch_gpu
    def test_semantic_hunyuan_pipeline_integration(self):
        # use the real model for integration test
        pipe = SemanticHunyuanDiTPipeline.from_pretrained("XCLiu/HunyuanDiT-0523", revision="refs/pr/2", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        prompt = "一个宇航员在骑马"
        generator = torch.Generator("cpu").manual_seed(0)

        image = pipe(
            prompt=prompt, height=1024, width=1024, generator=generator, num_inference_steps=2, output_type="np"
        ).images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([
            0.48388672, 0.33789062, 0.30737305,
            0.47875977, 0.25097656, 0.30029297,
            0.4440918, 0.26953125, 0.30078125
        ])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_save_load_optional_components(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs).images

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs).images

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, 1e-4)
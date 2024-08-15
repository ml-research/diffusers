import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

from ...pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from ...schedulers import DDPMScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, HunyuanDiT2DModel
from ...models.embeddings import get_2d_rotary_pos_embed
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...pipelines.semantic_stable_diffusion import SemanticStableDiffusionPipelineOutput
from ...callbacks import MultiPipelineCallbacks, PipelineCallback

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import HunyuanDiTSEGAPipeline

        >>> pipe = HunyuanDiTSEGAPipeline.from_pretrained(
        ...     "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Example: Cityscape
        >>> prompt = "A cityscape"
        >>> editing_prompt = ["crowed", "trees", "birds"]
        >>> image = pipe(prompt, editing_prompt=editing_prompt).images[0]
        ```
"""

STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]


# Copied from diffusers.pipelines.hunyuandits.pipeline_hunyuandits.map_to_standard_shapes
def map_to_standard_shapes(target_width, target_height):
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height


# Copied from diffusers.pipelines.hunyuandits.pipeline_hunyuandits.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.hunyuandits.pipeline_hunyuandits.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# based on diffusers.pipelines.hunyuandits.pipeline_hunyuandits.HunyuanDiTPipeline
class SemanticHunyuanDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using HunyuanDiT with SEGA (Semantic Guidance) latent editing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.BertModel`]):
            Frozen text-encoder. HunyuanDiT uses a fine-tuned [bilingual CLIP].
        tokenizer ([`~transformers.BertTokenizer`]):
            A `BertTokenizer` to tokenize text.
        transformer ([`HunyuanDiT2DModel`]):
            The HunyuanDiT model designed by Tencent Hunyuan.
        text_encoder_2 ([`~transformers.T5EncoderModel`]):
            The mT5 embedder. Specifically, it is 't5-v1_1-xxl'.
        tokenizer_2 ([`~transformers.MT5Tokenizer`]):
            The tokenizer for the mT5 embedder.
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with HunyuanDiT to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images for the safety checker.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "text_encoder_2",
        "tokenizer_2",
        "text_encoder",
        "tokenizer",
    ]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
    ]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: BertModel,
            tokenizer: BertTokenizer,
            transformer: HunyuanDiT2DModel,
            scheduler: DDPMScheduler,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
            text_encoder_2=T5EncoderModel,
            tokenizer_2=MT5Tokenizer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.default_sample_size = self.transformer.config.sample_size

    def encode_prompt(
            self,
            prompt: str,
            editing_prompt: Optional[Union[str, list]],
            device: torch.device = None,
            dtype: torch.dtype = None,
            num_images_per_prompt: int = 1,
            text_encoder_index: int = 0,
            max_sequence_length: Optional[int] = None,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            editing_prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            editing_prompt_attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ):
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 77
            if text_encoder_index == 1:
                max_length = 256
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            )[0]
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if editing_prompt is not None:
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            if isinstance(editing_prompt, list):
                editing_prompt_embeds = []
                editing_prompt_attention_masks = []

                for single_prompt in editing_prompt:
                    single_prompt_input = tokenizer(
                        single_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )

                    single_prompt_input_ids = single_prompt_input.input_ids
                    single_prompt_attention_mask = single_prompt_input.attention_mask.to(device)

                    single_prompt_embed = text_encoder(
                        single_prompt_input_ids.to(device),
                        attention_mask=single_prompt_attention_mask,
                    )[0]

                    editing_prompt_embeds.append(single_prompt_embed.to(dtype=dtype, device=device))
                    editing_prompt_attention_masks.append(single_prompt_attention_mask)

                # Convert lists to tensors
                editing_prompt_embeds = torch.cat(editing_prompt_embeds)
                editing_prompt_attention_mask = torch.cat(editing_prompt_attention_masks)
            else:
                raise ValueError(
                    f"`editing_prompt` has to be of type `str` or `list` but is {type(editing_prompt)}"
                )

        else:
            editing_prompt_embeds = None

        if editing_prompt_embeds is not None:
            editing_prompt_embeds = editing_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            editing_prompt_embeds = editing_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, editing_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask, editing_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if prompt_embeds_2 is not None and prompt_attention_mask_2 is None:
            raise ValueError("Must provide `prompt_attention_mask_2` when specifying `prompt_embeds_2`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if negative_prompt_embeds_2 is not None and negative_prompt_attention_mask_2 is None:
            raise ValueError(
                "Must provide `negative_prompt_attention_mask_2` when specifying `negative_prompt_embeds_2`."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if prompt_embeds_2 is not None and negative_prompt_embeds_2 is not None:
            if prompt_embeds_2.shape != negative_prompt_embeds_2.shape:
                raise ValueError(
                    "`prompt_embeds_2` and `negative_prompt_embeds_2` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_2` {prompt_embeds_2.shape} != `negative_prompt_embeds_2`"
                    f" {negative_prompt_embeds_2.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            prompt_embeds_2: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds_2: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            prompt_attention_mask_2: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = (1024, 1024),
            target_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            use_resolution_binning: bool = True,
            editing_prompt: Optional[Union[str, List[str]]] = None,
            editing_prompt_embeds: Optional[torch.Tensor] = None,
            editing_prompt_embeds_2: Optional[torch.Tensor] = None,
            editing_prompt_attention_mask: Optional[torch.Tensor] = None,
            editing_prompt_attention_mask_2: Optional[torch.Tensor] = None,
            reverse_editing_direction: bool = False,
            edit_guidance_scale: float = 5.0,
            edit_warmup_steps: int = 10,
            edit_cooldown_steps: Optional[int] = None,
            edit_threshold: float = 0.9,
            edit_momentum_scale: float = 0.1,
            edit_mom_beta: float = 0.4,
            edit_weights: Optional[List[float]] = None,
            sem_guidance: Optional[List[torch.Tensor]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation with HunyuanDiT and SeGa.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*):
                The height in pixels of the generated image. If not provided, it defaults to the model's default sample size.
            width (`int`, *optional*):
                The width in pixels of the generated image. If not provided, it defaults to the model's default sample size.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
                images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings for the second text encoder (T5).
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_embeds_2 (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings for the second text encoder (T5).
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt.
            prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the prompt for the second text encoder (T5).
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt.
            negative_prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt for the second text encoder (T5).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that will be called at the end of each denoising step during the inference. The function will be
                called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*, defaults to `["latents"]`):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will
                be passed as `callback_kwargs` argument.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int, int]`, *optional*, defaults to (1024, 1024)):
                Original size of the image before any resizing is done.
            target_size (`Tuple[int, int]`, *optional*):
                Target size for the image. If not specified, it will be set to (height, width).
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to (0, 0)):
                Coordinates for the top-left crop of the image.
            use_resolution_binning (`bool`, *optional*, defaults to True):
                Whether to use resolution binning. If True, the input resolution will be mapped to the closest
                standard resolution.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for semantic guidance (editing).
            editing_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings for editing prompt. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            editing_prompt_embeds_2 (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings for editing prompt for the second text encoder (T5).
            editing_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the editing prompt.
            editing_prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the editing prompt for the second text encoder (T5).
            reverse_editing_direction (`bool`, *optional*, defaults to False):
                Whether to reverse the editing direction. If True, the model will try to reduce the attributes specified
                in the editing prompt.
            edit_guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale for semantic guidance. Higher values guide the image more towards the editing prompt.
            edit_warmup_steps (`int`, *optional*, defaults to 10):
                Number of warmup steps for semantic guidance.
            edit_cooldown_steps (`int`, *optional*, defaults to None):
                Number of cooldown steps for semantic guidance. If None, semantic guidance is applied until the end.
            edit_threshold (`float`, *optional*, defaults to 0.9):
                Threshold for semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Momentum beta for semantic guidance.
            edit_weights (`List[float]`, *optional*):
                Weights for each editing prompt. If not provided, all editing prompts are weighted equally.
            sem_guidance (`List[torch.FloatTensor]`, *optional*):
                Pre-generated semantic guidance. If provided, it will be used instead of generating it from editing_prompt.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeds is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_embeds.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False
        
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode prompt and editing prompt
        (prompt_embeds, negative_prompt_embeds, editing_prompt_embeds,
         prompt_attention_mask, negative_prompt_attention_mask, editing_prompt_attention_mask
         ) = self.encode_prompt(
            prompt,
            editing_prompt,
            device,
            self.transformer.dtype,
            num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            editing_prompt_embeds=editing_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            editing_prompt_attention_mask=None,
            max_sequence_length=77,
            text_encoder_index=0,
        )
        (prompt_embeds_2, negative_prompt_embeds_2, editing_prompt_embeds_2,
         prompt_attention_mask_2, negative_prompt_attention_mask_2, editing_prompt_attention_mask_2
         ) = self.encode_prompt(
            prompt,
            editing_prompt,
            device,
            torch.float32,
            num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds_2,
            editing_prompt_embeds=editing_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask_2,
            editing_prompt_attention_mask=None,
            max_sequence_length=256,
            text_encoder_index=1,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs for SEGA
        extra_step_kwargs = self.prepare_extra_step_kwargs(
            generator, eta
            # , editing_prompt_embeds, reverse_editing_direction, edit_guidance_scale,
            # edit_warmup_steps, edit_cooldown_steps, edit_threshold, edit_momentum_scale, edit_mom_beta,
            # edit_weights
        )

        edit_momentum = None

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None

        # 7 create image_rotary_emb, style embedding & time ids
        grid_height = height // 8 // self.transformer.config.patch_size
        grid_width = width // 8 // self.transformer.config.patch_size
        base_size = 512 // 8 // self.transformer.config.patch_size
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads, grid_crops_coords, (grid_height, grid_width)
        )

        style = torch.tensor([0], device=device)

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        if do_classifier_free_guidance:
            if enable_edit_guidance:
                editing_prompt_embeds = editing_prompt_embeds.reshape([-1, *tuple(prompt_embeds.shape[1:])])
                editing_prompt_embeds_2 = editing_prompt_embeds_2.reshape([-1, *tuple(prompt_embeds_2.shape[1:])])
                editing_prompt_attention_mask = editing_prompt_attention_mask.reshape([-1, *tuple(prompt_attention_mask.shape[1:])])
                editing_prompt_attention_mask_2 = editing_prompt_attention_mask_2.reshape([-1, *tuple(prompt_attention_mask_2.shape[1:])])
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, editing_prompt_embeds])
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, editing_prompt_attention_mask])
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2, editing_prompt_embeds_2])
                prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2, editing_prompt_attention_mask_2])
                add_time_ids = torch.cat([add_time_ids] * (2+enabled_editing_prompts), dim=0)  # Updated to 3 because we added editing_prompt
                style = torch.cat([style] * (2+enabled_editing_prompts), dim=0)  # Updated to 3 because we added editing_prompt
            else:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, ])
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2, ])
                prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2,])
                add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
                style = torch.cat([style] * 2, dim=0)

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device)
        prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
        add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=device).repeat(
            batch_size * num_images_per_prompt, 1
        )
        style = style.to(device=device).repeat(batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * (2 +enabled_editing_prompts)) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )

                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                noise_pred, _ = noise_pred.chunk(2, dim=1)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                    noise_pred_edit_concepts = noise_pred_out[2:]

                    # default text guidance
                    noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # noise_guidance = (noise_pred_text - noise_pred_edit_concepts[0])

                    if self.uncond_estimates is None:
                        self.uncond_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_uncond.shape))
                    self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                    if self.text_estimates is None:
                        self.text_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))
                    self.text_estimates[i] = noise_pred_text.detach().cpu()

                    if self.edit_estimates is None and enable_edit_guidance:
                        self.edit_estimates = torch.zeros(
                            (num_inference_steps + 1, len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )

                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))

                    if edit_momentum is None:
                        edit_momentum = torch.zeros_like(noise_guidance)

                    if enable_edit_guidance:
                        concept_weights = torch.zeros(
                            (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                            device=self.device,
                            dtype=noise_guidance.dtype,
                        )
                        noise_guidance_edit = torch.zeros(
                            (len(noise_pred_edit_concepts), *noise_guidance.shape),
                            device=self.device,
                            dtype=noise_guidance.dtype,
                        )
                        # noise_guidance_edit = torch.zeros_like(noise_guidance)
                        warmup_inds = []
                        for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                            self.edit_estimates[i, c] = noise_pred_edit_concept
                            if isinstance(edit_guidance_scale, list):
                                edit_guidance_scale_c = edit_guidance_scale[c]
                            else:
                                edit_guidance_scale_c = edit_guidance_scale

                            if isinstance(edit_threshold, list):
                                edit_threshold_c = edit_threshold[c]
                            else:
                                edit_threshold_c = edit_threshold
                            if isinstance(reverse_editing_direction, list):
                                reverse_editing_direction_c = reverse_editing_direction[c]
                            else:
                                reverse_editing_direction_c = reverse_editing_direction
                            if edit_weights:
                                edit_weight_c = edit_weights[c]
                            else:
                                edit_weight_c = 1.0
                            if isinstance(edit_warmup_steps, list):
                                edit_warmup_steps_c = edit_warmup_steps[c]
                            else:
                                edit_warmup_steps_c = edit_warmup_steps

                            if isinstance(edit_cooldown_steps, list):
                                edit_cooldown_steps_c = edit_cooldown_steps[c]
                            elif edit_cooldown_steps is None:
                                edit_cooldown_steps_c = i + 1
                            else:
                                edit_cooldown_steps_c = edit_cooldown_steps
                            if i >= edit_warmup_steps_c:
                                warmup_inds.append(c)
                            if i >= edit_cooldown_steps_c:
                                noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                                continue

                            noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                            # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                            tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                            tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                            if reverse_editing_direction_c:
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                            concept_weights[c, :] = tmp_weights

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp.dtype == torch.float32:
                                tmp = torch.quantile(
                                    torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp.dtype)

                            noise_guidance_edit_tmp = torch.where(
                                torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                                noise_guidance_edit_tmp,
                                torch.zeros_like(noise_guidance_edit_tmp),
                            )
                            noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                            # noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

                        warmup_inds = torch.tensor(warmup_inds).to(self.device)
                        if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                            concept_weights = concept_weights.to("cpu")  # Offload to cpu
                            noise_guidance_edit = noise_guidance_edit.to("cpu")

                            concept_weights_tmp = torch.index_select(concept_weights.to(self.device), 0, warmup_inds)
                            concept_weights_tmp = torch.where(
                                concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                            )
                            concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                            # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                            noise_guidance_edit_tmp = torch.index_select(
                                noise_guidance_edit.to(self.device), 0, warmup_inds
                            )
                            noise_guidance_edit_tmp = torch.einsum(
                                "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                            )
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp
                            noise_guidance = noise_guidance + noise_guidance_edit_tmp

                            self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                            del noise_guidance_edit_tmp
                            del concept_weights_tmp
                            concept_weights = concept_weights.to(self.device)
                            noise_guidance_edit = noise_guidance_edit.to(self.device)

                        concept_weights = torch.where(
                            concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                        )

                        concept_weights = torch.nan_to_num(concept_weights)

                        noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                        noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                        edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                        if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                            noise_guidance = noise_guidance + noise_guidance_edit
                            self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                    if sem_guidance is not None:
                        edit_guidance = sem_guidance[i].to(self.device)
                        noise_guidance = noise_guidance + edit_guidance

                    noise_pred = noise_pred_uncond + noise_guidance
                    # compute the previous noisy sample x_t -> x_t-1

                    # perform guidance
                    # if do_classifier_free_guidance:
                    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    noise_pred = noise_pred_uncond + noise_guidance
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

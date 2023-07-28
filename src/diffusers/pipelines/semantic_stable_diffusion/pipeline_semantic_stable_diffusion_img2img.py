import inspect
import warnings
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...image_processor import VaeImageProcessor
from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...schedulers import KarrasDiffusionSchedulers, DDIMInverseScheduler
from ...utils import deprecate, logging
from . import SemanticStableDiffusionPipelineOutput

import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.optim.adam import Adam
import torch.nn.functional as nnf


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class SemanticStableDiffusionImg2ImgPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation with latent editing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    This model builds on the implementation of ['StableDiffusionPipeline']

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            inverse_scheduler: DDIMInverseScheduler,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

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

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            inverse_scheduler=inverse_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        return extra_step_kwargs

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        height,
        width,
        callback_steps,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
            self,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            editing_prompt: Optional[Union[str, List[str]]] = None,
            editing_prompt_embeddings: Optional[torch.FloatTensor] = None,
            reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
            edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
            edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
            edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
            edit_threshold: Optional[Union[float, List[float]]] = 0.9,
            edit_momentum_scale: Optional[float] = 0.1,
            edit_mom_beta: Optional[float] = 0.4,
            edit_weights: Optional[List[float]] = None,
            sem_guidance: Optional[List[torch.FloatTensor]] = None,
            PnP: bool = True,
            pnp_f_strength: float = 0.8,
            pnp_attn_strength: float = 0.5,
            **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for Semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.FloatTensor`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as list values should correspond to `editing_prompt`.
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance will not be applied. Momentum
                will still be calculated for those steps and applied once all warmup periods are over.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0
                momentum will be disabled. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `sld_warmup_steps`. Momentum will only be added to latent guidance once all warmup periods are
                finished.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `edit_warmup_steps`.
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally.
            sem_guidance (`List[torch.FloatTensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Returns:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] if `return_dict` is True,
            otherwise a `tuple. When returning a tuple, the first element is a list with the generated images, and the
            second element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """

        latents = self.init_latents
        batch_size = 1

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(height, width, callback_steps)

        # 2. Define call parameters
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

            if PnP:
                # there is no prediction for which features can be injected
                logger.warning("PnP is disabled because PnP requires at least one editing prompt.")
                PnP = False

        if enable_edit_guidance:
            # get safety text embeddings
            if editing_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                edit_concepts_input_ids = edit_concepts_input.input_ids
                untruncated_ids = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="longest",
                    return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= edit_concepts_input_ids.shape[-1] and not torch.equal(
                    edit_concepts_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                edit_concepts = self.text_encoder(edit_concepts_input_ids.to(self.device))[0]
            else:
                edit_concepts = editing_prompt_embeddings.to(self.device).repeat(batch_size, 1, 1)

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        uncond_embeddings = self.get_uncond_embeddings()

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For semantic guidance, we need to do multiple forward passes.
        # Here we concatenate the unconditional and concept text embeddings into a single batch
        # to avoid doing multiple forward passes
        if enable_edit_guidance:
            text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
        else:
            text_embeddings = torch.cat([uncond_embeddings])

        if PnP:
            # Plug-and_Play specific
            uncond_tokens = [""]

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            pnp_guidance_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            seq_len = pnp_guidance_embeds.shape[1]
            pnp_guidance_embeds = pnp_guidance_embeds.repeat(batch_size, num_images_per_prompt, 1)
            pnp_guidance_embeds = pnp_guidance_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # 4. Prepare timesteps
        #self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if PnP:
            pnp_f_t = int(len(timesteps) * pnp_f_strength)
            pnp_attn_t = int(len(timesteps) * pnp_attn_strength)

            qk_injection_timesteps = self.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
            conv_injection_timesteps = self.scheduler.timesteps[:pnp_f_t] if pnp_f_t >= 0 else []

            register_attention_control_efficient(self, qk_injection_timesteps)
            register_conv_control_efficient(self, conv_injection_timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            self.device,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # Initialize edit_momentum to None
        edit_momentum = None
        self.sem_guidance = None

        for i, t in enumerate(self.progress_bar(timesteps)):
            source_latents = self.latents_path[i]

            # expand the latents if we are doing semantic guidance, to avoid doing multiple forward passes
            latent_model_input = (
                torch.cat([source_latents] + [latents] * (1 + self.enabled_editing_prompts))
                if PnP
                else torch.cat([latents] * (1 + self.enabled_editing_prompts))
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if PnP:
                # register the time step and features in pnp injection modules
                register_time(self, t.item())

            text_embed_input = (
                torch.cat([pnp_guidance_embeds, text_embeddings], dim=0)
                if PnP
                else text_embeddings
            )

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample

            # perform semantic guidance
            if enable_edit_guidance:
                if PnP:
                    noise_pred_out = noise_pred.chunk(2 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond = noise_pred_out[1]
                    noise_pred_edit_concepts = noise_pred_out[2:]
                else:
                    noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond = noise_pred_out[0]
                    noise_pred_edit_concepts = noise_pred_out[1:]

                noise_guidance = torch.zeros_like(noise_pred_uncond)

                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros((self.num_inversion_steps+1, *noise_pred_uncond.shape))

                if edit_momentum is None:
                    edit_momentum = torch.zeros_like(noise_guidance)

                concept_weights = torch.zeros(
                    (len(noise_pred_edit_concepts), noise_guidance.shape[0]), device=self.device
                )
                noise_guidance_edit = torch.zeros(
                    (len(noise_pred_edit_concepts), *noise_guidance.shape), device=self.device
                )
                # noise_guidance_edit = torch.zeros_like(noise_guidance)
                warmup_inds = []
                for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):

                    # 1. get SEGA parameters for concept
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

                    # 2. no semantic guidance for cooldown steps
                    if i >= edit_cooldown_steps_c:
                        noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                        continue

                    # 3. semantic guidance term
                    noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond

                    if reverse_editing_direction_c:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1

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

                    # 4. weighting factor for different concepts
                    concept_weights[c, :] = torch.full((1,noise_guidance.shape[0]), edit_weight_c) # * (1 / enabled_editing_prompts)

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
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return SemanticStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def get_uncond_embeddings(self):
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        return uncond_embeddings

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def ddim_loop(self, latent, skip_first_step):
        uncond_embeddings = self.get_uncond_embeddings()
        all_latent = [latent]
        latent = latent.clone().detach()
        noise_preds = []

        timesteps = self.inverse_scheduler.timesteps
        for i, t in enumerate(timesteps):
            if i == 0 and skip_first_step:
                # skipping the first step is beneficial for DDIM inversion
                continue

            # predict the noise residual
            noise_pred = self.unet(latent, t, encoder_hidden_states=uncond_embeddings)["sample"]

            # compute the next noisy sample x_t -> x_t+1
            latent = self.inverse_scheduler.step(noise_pred, t, latent).prev_sample

            all_latent.append(latent.detach().clone())
            noise_preds.append(noise_pred.detach().clone())
        return all_latent, noise_preds

    @torch.no_grad()
    def ddim_inversion(self, image_path, offsets, skip_first_step):
        image_gt = load_512(image_path, *offsets)
        latent = self.image2latent(image_gt)
        image_rec = self.decode_latents(latent)
        ddim_latents, noise_preds = self.ddim_loop(latent, skip_first_step)
        return image_gt, image_rec, ddim_latents, noise_preds

    def invert(self, image_path: str, skip_first_step: bool, offsets=(0, 0, 0, 0), verbose=False,
               num_inverstion_steps: int = 50, return_all_latents=False, return_noise_preds=False):

        self.num_inversion_steps = num_inverstion_steps
        self.scheduler.set_timesteps(self.num_inversion_steps)
        self.inverse_scheduler.set_timesteps(self.num_inversion_steps)

        if verbose:
            print("DDIM inversion...")
        image_gt, image_rec, ddim_latents, noise_preds = self.ddim_inversion(image_path, offsets, skip_first_step)

        self.init_latents = ddim_latents[-1]
        latents_path = ddim_latents.copy()
        latents_path.reverse()
        self.latents_path = latents_path

        if return_all_latents:
            return (image_gt, image_rec), ddim_latents
        if return_noise_preds:
            return noise_preds
        return (image_gt, image_rec), ddim_latents[-1]

# Copied and slightly modified from https://github.com/MichalGeyer/pnp-diffusers
def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        # modified
        editing_prompts = model.enabled_editing_prompts

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // (2 + editing_prompts))

                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]

                if editing_prompts > 0:
                    # inject editing
                    for i in range(editing_prompts):
                        q[(2+i) * source_batch_size: (3+i) * source_batch_size] = q[:source_batch_size]
                        k[(2+i) * source_batch_size: (3+i) * source_batch_size] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

# Copied and slightly modified from https://github.com/MichalGeyer/pnp-diffusers
def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        # modified
        editing_prompts = model.enabled_editing_prompts

        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // (2 + editing_prompts))

                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:3 * source_batch_size] = hidden_states[:source_batch_size]

                if editing_prompts > 0:
                    # inject editing
                    for i in range(editing_prompts):
                        hidden_states[(3+i) * source_batch_size:(4+i) * source_batch_size] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

# Copied from https://github.com/MichalGeyer/pnp-diffusers
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

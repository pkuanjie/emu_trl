# Copyright 2023 DDPO-pytorch authors (Kevin Black), The HuggingFace Team, metric-space. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.utils import convert_state_dict_to_diffusers

from ..core import randn_tensor
from ..import_utils import is_peft_available
from .Emu1.models.pipeline import EmuGenerationPipeline


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from peft import get_peft_model, prepare_model_for_kbit_training


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value


@dataclass
class DDPOEmu1LMMPipelineOutput(object):
    """
    Output class for the diffusers pipeline of emu to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`List[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`List[torch.Tensor]`):
            The log probabilities of the latents.

    """

    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor


@dataclass
class DDPOEmu1LMMSchedulerOutput(object):
    """
    Output class for the diffusers scheduler of emu to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """

    latents: torch.Tensor
    log_probs: torch.Tensor


class DDPOEmu1LMMPipeline(object):
    """
    Main class for the diffusers pipeline of emu to be finetuned with the DDPO trainer
    """

    def __call__(self, *args, **kwargs) -> DDPOEmu1LMMPipelineOutput:
        raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOEmu1LMMSchedulerOutput:
        raise NotImplementedError

    @property
    def unet(self):
        """
        Returns the 2d U-Net model used for diffusion.
        """
        raise NotImplementedError

    @property
    def vae(self):
        """
        Returns the Variational Autoencoder model used from mapping images to and from the latent space
        """
        raise NotImplementedError

    @property
    def tokenizer(self):
        """
        Returns the tokenizer used for tokenizing text inputs
        """
        raise NotImplementedError

    @property
    def scheduler(self):
        """
        Returns the scheduler associated with the pipeline used for the diffusion process
        """
        raise NotImplementedError

    @property
    def multimodal_encoder(self):
        """
        Returns the text encoder used for encoding text inputs
        """
        raise NotImplementedError

    @property
    def autocast(self):
        """
        Returns the autocast context manager
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        """
        Saves all of the model weights
        """
        raise NotImplementedError

    def get_trainable_layers(self, *args, **kwargs):
        """
        Returns the trainable parameters of the pipeline
        """
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_save_state_pre_hook which is run before saving state
        """
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_lad_state_pre_hook which is run before loading state
        """
        raise NotImplementedError


def _left_broadcast(input_tensor, shape):
    """
    As opposed to the default direction of broadcasting (right to left), this function broadcasts
    from left to right
        Args:
            input_tensor (`torch.FloatTensor`): is the tensor to broadcast
            shape (`Tuple[int]`): is the shape to broadcast to
    """
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError(
            "The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to"
        )
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> DDPOEmu1LMMSchedulerOutput:
    """

    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)

    Returns:
        `DDPOSchedulerOutput`: the predicted sample at the previous timestep and the log probability of the sample
    """

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return DDPOEmu1LMMSchedulerOutput(prev_sample.type(sample.dtype), log_prob)


# 1. The output type for call is different as the logprobs are now returned
# 2. An extra method called `scheduler_step` is added which is used to constraint the scheduler output
@torch.no_grad()
def pipeline_step(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    crop_info: List[int] = [0, 0],
    original_size: List[int] = [1024, 1024],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.  Args: prompt (`str` or `List[str]`, *optional*): The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.  instead.  height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor): The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
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
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
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
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        `DDPOPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    # self.check_inputs(
    #     prompt,
    #     height,
    #     width,
    #     callback_steps,
    #     negative_prompt,
    #     prompt_embeds,
    #     negative_prompt_embeds,
    # )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    dtype = self.dtype(self.unet)

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._prepare_prompt_embed(
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    shape = (
        batch_size,
        self.unet.config.in_channels,
        height // self.vae_scale_factor,
        width // self.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=dtype)
    latents = latents * self.scheduler.init_noise_sigma

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler_output = scheduler_step(self.scheduler, noise_pred, t, latents, eta)
        latents = scheduler_output.latents
        log_prob = scheduler_output.log_probs

        all_latents.append(latents)
        all_log_probs.append(log_prob)

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        # image, has_nsfw_concept = self.run_safety_checker(image, device=device, dtype=dtype)
        has_nsfw_concept = None
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    # if not output_type == "latent":
    #     image = self.decode_latents(latents)
    #     image, has_nsfw_concept = self.run_safety_checker(image)
    # else:
    #     image = latents
    #     has_nsfw_concept = None

    # image = self.numpy_to_pil(image)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return DDPOEmu1LMMPipelineOutput(image, all_latents, all_log_probs), (cross_attention_kwargs,)


class DefaultDDPOEmu1LMMPipeline(DDPOEmu1LMMPipeline):
    def __init__(self, pretrained_model_name: str, *, use_lora: bool = True):
        emu1_args = {
            "ckpt-path": pretrained_model_name,
            "instruct": False,
        }
        emu1_args = AttrDict(emu1_args)
        self.emu1_pipeline = EmuGenerationPipeline.from_pretrained(
            path=pretrained_model_name,
            args=emu1_args,
        )

        self.use_lora = use_lora

        try:
            self.emu1_pipeline.load_lora_weights(
                pretrained_model_name,
                weight_name="pytorch_lora_weights.safetensors",
            )
            self.use_lora = True
        except OSError:
            if use_lora:
                warnings.warn(
                    "If you are aware that the pretrained model has no lora weights to it, ignore this message. "
                    "Otherwise please check the if `pytorch_lora_weights.safetensors` exists in the model folder."
                )

        self.emu1_pipeline.scheduler = DDIMScheduler.from_config(self.emu1_pipeline.scheduler.config)

        # memory optimization
        self.emu1_pipeline.vae.requires_grad_(False)
        self.emu1_pipeline.unet.requires_grad_(False)
        self.emu1_pipeline.emu_encoder.requires_grad_(not self.use_lora)
        self.negative_prompt = {}

    def __call__(self, *args, **kwargs) -> DDPOEmu1LMMPipelineOutput:
        return pipeline_step(self.emu1_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs) -> DDPOEmu1LMMSchedulerOutput:
        return scheduler_step(self.emu1_pipeline.scheduler, *args, **kwargs)

    @property
    def unet(self):
        return self.emu1_pipeline.unet

    @property
    def vae(self):
        return self.emu1_pipeline.vae

    @property
    def tokenizer(self):
        return self.emu1_pipeline.tokenizer

    @property
    def scheduler(self):
        return self.emu1_pipeline.scheduler

    @property
    def emu_encoder(self):
        return self.emu1_pipeline.emu_encoder

    # added for ddpo
    def _get_negative_prompt_embedding(self, key: str):
        if key not in self.negative_prompt:
            self.negative_prompt[key] = self.emu_encoder.generate_image(text=[key])
        return self.negative_prompt[key]

    @property
    def autocast(self):
        return contextlib.nullcontext if self.use_lora else None

    def save_pretrained(self, output_dir):
        if self.use_lora:
            state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(self.emu1_pipeline.emu_encoder.decoder.lm)
            )
            self.emu1_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        self.emu1_pipeline.save_pretrained(output_dir)

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(
                r=1,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                # task_type="CAUSAL_LM",
            )
            # print_trainable_parameters(self.emu1_pipeline.emu_encoder.decoder.lm.model)

            self.emu1_pipeline.emu_encoder.decoder.lm.model = prepare_model_for_kbit_training(
                self.emu1_pipeline.emu_encoder.decoder.lm.model, use_gradient_checkpointing=True
            )

            self.emu1_pipeline.emu_encoder.decoder.lm.model = get_peft_model(
                self.emu1_pipeline.emu_encoder.decoder.lm.model, lora_config
            )
            # print_trainable_parameters(self.emu1_pipeline.emu_encoder.decoder.lm.model)

            # To avoid accelerate unscaling problems in FP16.
            # for param in self.emu1_pipeline.emu_encoder.decoder.lm.model.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            # if param.requires_grad:
            #     param.data = param.to(torch.float32)
            return self.emu1_pipeline.emu_encoder.decoder.lm.model
        else:
            return self.emu1_pipeline.emu_encoder.decoder.lm.model

    def save_checkpoint(self, models, weights, output_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora and hasattr(models[0], "peft_config") and getattr(models[0], "peft_config", None) is not None:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            self.emu1_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")

    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora:
            lora_state_dict, network_alphas = self.emu1_pipeline.lora_state_dict(
                input_dir, weight_name="pytorch_lora_weights.safetensors"
            )
            self.emu1_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])

        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")

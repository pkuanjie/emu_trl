# -*- coding: utf-8 -*-

import os
import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union, Tuple

import torch
import torch.nn as nn
from torchvision import transforms as TF
from diffusers.loaders import LoraLoaderMixin

from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor

from .modeling_emu import Emu


class EmuGenerationPipeline(nn.Module, LoraLoaderMixin):

    def __init__(
        self,
        multimodal_model: str,
        feature_extractor: str,
        safety_checker: str,
        scheduler: str,
        unet: str,
        vae: str,
        eva_size=224,
        eva_mean=(0.48145466, 0.4578275, 0.40821073),
        eva_std=(0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(
            unet,
        )
        self.vae = AutoencoderKL.from_pretrained(
            vae,
        )
        self.scheduler = PNDMScheduler.from_pretrained(
            scheduler,
        )

        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     safety_checker,
        # )
        self.safety_checker = None
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            feature_extractor,
        )

        self.emu_encoder = self.prepare_emu("Emu-14B", multimodal_model, **kwargs)
        self.tokenizer = self.emu_encoder.decoder.tokenizer

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.eval()

        self.transform = TF.Compose(
            [
                TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
                TF.Normalize(mean=eva_mean, std=eva_std),
            ]
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # @torch.no_grad()
    def forward(
        self,
        inputs: List[Union[Image.Image, str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Tuple[Image.Image, bool]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        batch_size = 1
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        # Bx4xHxW
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)

        # 4. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, dtype)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
        return image[0], has_nsfw_concept[0] if has_nsfw_concept is not None else has_nsfw_concept

    # newly added for ddpo
    def _prepare_prompt_embed(
        self,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds,
        negative_prompt_embeds,
        lora_scale=None,  # a dummy argument
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        batch_size = prompt_embeds.shape[0]

        if self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # concatenate for backwards comp
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # newly added for ddpo
    @property
    def _execution_device(self):
        return self.device(self.unet)

    def device(self, module):
        return next(module.parameters()).device

    def dtype(self, module):
        return next(module.parameters()).dtype

    def _prepare_and_encode_inputs(
        self,
        inputs: List[Union[str, Image.Image]],
        device: torch.device = "cpu",
        dtype: str = torch.float32,
        do_classifier_free_guidance: bool = False,
        placeholder: str = "[<IMG_PLH>]",
    ) -> torch.Tensor:
        text_prompt = ""
        image_prompt = []
        for x in inputs:
            if isinstance(x, str):
                text_prompt += x
            else:
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        # Nx3x224x224
        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if do_classifier_free_guidance:
            text_prompt = [text_prompt, ""]
        else:
            text_prompt = [text_prompt]

        prompt = self.emu_encoder.generate_image(
            text=text_prompt,
            image=image_prompt,
            placeholder=placeholder,
        )

        return prompt

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")

        if images.shape[-1] != 1 and images.shape[-1] != 3:
            images = np.transpose(images, (0, 2, 3, 1))

        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(
        self,
        image: List[Image.Image],
        device: str,
        dtype: str,
    ):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image.cpu().data.numpy()), return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def prepare_emu(
        self,
        model_name: str,
        model_path: str,
        **kwargs,
    ) -> nn.Module:
        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_path, f"{model_name}.json"), "r", encoding="utf8") as f:
            model_cfg = json.load(f)

        model = Emu(**model_cfg, cast_dtype=torch.float, **kwargs)
        ckpt = torch.load(model_path, map_location="cpu")
        if "module" in ckpt:
            model.load_state_dict(ckpt["module"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

        return model

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        multimodal_model = kwargs.pop("multimodal_model", None)
        feature_extractor = kwargs.pop("feature_extractor", None)
        safety_checker = kwargs.pop("safety_checker", None)
        scheduler = kwargs.pop("scheduler", None)
        unet = kwargs.pop("unet", None)
        vae = kwargs.pop("vae", None)

        check_if_none = lambda x, y: y if x is None else x

        multimodal_model = check_if_none(multimodal_model, f"{path}/multimodal_encoder/pytorch_model.bin")
        feature_extractor = check_if_none(feature_extractor, f"{path}/feature_extractor")
        safety_checker = check_if_none(safety_checker, f"{path}/safety_checker")
        scheduler = check_if_none(scheduler, f"{path}/scheduler")
        unet = check_if_none(unet, f"{path}/unet")
        vae = check_if_none(vae, f"{path}/vae")

        return cls(
            multimodal_model=multimodal_model,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            **kwargs,
        )

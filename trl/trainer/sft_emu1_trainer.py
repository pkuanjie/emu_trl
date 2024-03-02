# Copyright 2023 DDPO-pytorch authors (Kevin Black), metric-space, The HuggingFace Team. All rights reserved.
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

import os
from datetime import datetime
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, PNDMScheduler
from huggingface_hub import whoami
from diffusers.training_utils import compute_snr

from ..models import DDPOStableDiffusionPipeline, DDPOEmu1Pipeline
from . import BaseTrainer, DDPOConfig, SFTConfig
from .utils import PerPromptStatTracker
import logging
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- ddpo
- diffusers
- reinforcement-learning
- text-to-image
- stable-diffusion
---

# {model_name}

This is a diffusion model that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for image generation conditioned with text.

"""


def log_with_time(message):
    c = datetime.now()
    print(f"time: {c} | {message}")


class SFTEmu1Trainer(BaseTrainer):
    """
    The SFTEmuTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models.
    Note, this trainer is heavily inspired by the work here: https://github.com/kvablack/ddpo-pytorch
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        **config** (`SFTConfig`) -- Configuration object for SFTTrainer. Check the documentation of `PPOConfig` for more
         details.
        **reward_function** (Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]) -- Reward function to be used
        **prompt_function** (Callable[[], Tuple[str, Any]]) -- Function to generate prompts to guide model
        **sd_pipeline** (`SFTStableDiffusionPipeline`) -- Stable Diffusion pipeline to be used for training.
        **image_samples_hook** (Optional[Callable[[Any, Any, Any], Any]]) -- Hook to be called to log images
    """

    _tag_names = ["trl", "ddpo"]

    def __init__(
        self,
        config: SFTConfig,
        train_loader,
        test_prompts,
        sd_pipeline: DDPOEmu1Pipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.train_loader = train_loader
        self.test_prompts = test_prompts
        self.config = config
        self.image_samples_callback = image_samples_hook
        self.prediction_type = None
        self.snr_gamma = None

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1

        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        is_okay, message = self._config_check()
        if not is_okay:
            raise ValueError(message)

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(sft_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        # For mixed precision training we cast all non-trainable weights (vae, non-lora emu_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.inference_dtype = inference_dtype

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.emu_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        self.noise_scheduler = PNDMScheduler.from_config(self.sd_pipeline.emu1_pipeline.scheduler.config)

        trainable_layers = self.sd_pipeline.get_trainable_layers()
        trainable_layers.to(self.accelerator.device, dtype=inference_dtype)

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # emu2 has a more convenient way to get the negative prompt embed
        # self.neg_prompt_embed = self.sd_pipeline.emu_encoder(
        #     self.sd_pipeline.tokenizer(
        #         [""] if self.config.negative_prompts is None else self.config.negative_prompts,
        #         return_tensors="pt",
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.sd_pipeline.tokenizer.model_max_length,
        #     ).input_ids.to(self.accelerator.device)
        # )[0]
        self.neg_prompt_embed = self.sd_pipeline._get_negative_prompt_embedding("")

        # if config.per_prompt_stat_tracking:
        #     self.stat_tracker = PerPromptStatTracker(
        #         config.per_prompt_stat_tracking_buffer_size,
        #         config.per_prompt_stat_tracking_min_count,
        #     )

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet = self.accelerator.prepare(trainable_layers)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers = self.accelerator.prepare(trainable_layers)

        self.optimizer = self._setup_optimizer(
            self.trainable_layers.parameters()
            if not isinstance(self.trainable_layers, list)
            else self.trainable_layers
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

        del trainable_layers
        torch.cuda.empty_cache()

    # def compute_rewards(self, prompt_image_pairs, is_async=False):
    #     if not is_async:
    #         rewards = []
    #         for images, prompts, prompt_metadata in prompt_image_pairs:
    #             reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
    #             rewards.append(
    #                 (
    #                     torch.as_tensor(reward, device=self.accelerator.device),
    #                     reward_metadata,
    #                 )
    #             )
    #     else:
    #         rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
    #         rewards = [
    #             (torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result())
    #             for reward, reward_metadata in rewards
    #         ]
    #
    #     return zip(*rewards)
    #
    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.

        """

        if self.image_samples_callback is not None:
            self.sd_pipeline.unet.eval()
            image_prompt_list = self._generate_samples(self.config.sample_batch_size)

            if self.accelerator.is_main_process:
                self.image_samples_callback(image_prompt_list, global_step, self.accelerator.trackers[0])

        self.sd_pipeline.unet.train()
        for batch_idx, batched_samples in emunerate(self.train_loader):

            global_step = self._train_batched_samples(epoch, global_step, batched_samples)
            # ensure optimization step at the end of the inner epoch
            if not self.accelerator.sync_gradients:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )

        if epoch != 0 and epoch % self.config.save_freq == 0:
            self.accelerator.save_state()

        return global_step

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    @torch.no_grad()
    def _generate_samples(self, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (List[Dict[str, torch.Tensor]]), prompt_image_pairs (List[List[Any]])
        """
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()

        for eval_idx in range(len(self.test_prompts) // batch_size):
            this_prompts = self.test_prompts[eval_idx * batch_size : (eval_idx + 1) * batch_size]

            prompt_embeds = self.sd_pipeline.emu_encoder.generate_image_efficient(this_prompts, max_token_length=120)
            logger.info(f"Prompt embeds shape: {prompt_embeds.shape}")

            with self.autocast():
                sd_output, _ = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=0.0,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images

            logger.info(f"Images shape: {images.shape}")
            print(this_prompts[0])
            plt.imshow(images[0].cpu().numpy().transpose(1, 2, 0))
            plt.show()

            prompt_image_pairs.append([images, this_prompts])

        return prompt_image_pairs

    def _train_batched_samples(self, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (List[Dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """

        with self.accelerator.accumulate(self.sd_pipeline.unet):
            # Convert images to latent space
            latents = self.sd_pipeline.vae.encode(batch["image"].to(dtype=self.inference_dtype)).latent_dist.sample()
            latents = latents * self.sd_pipeline.emu_encoder.vae_scale_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            # prompt_embeds = self.sd_pipeline.emu_encoder.generate_image_efficient(batched_samples["prompt"])
            prompt_embeds = self.sd_pipeline.emu_encoder.teacher_forcing(batched_samples["prompt"])

            # Get the target for loss depending on the prediction type
            if self.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(prediction_type=self.prediction_type)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # # Prepare micro-conditions.
            # added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            # if getattr(transformer, "module", transformer).config.sample_size == 128:
            #     resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
            #     aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
            #     resolution = resolution.to(dtype=weight_dtype, device=latents.device)
            #     aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=latents.device)
            #     added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            # Predict the noise residual and compute loss
            model_pred = self.sd_pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample

            if self.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(self.noise_scheduler, timesteps)
                if self.noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                    torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    (
                        self.trainable_layers.parameters()
                        if not isinstance(self.trainable_layers, list)
                        else self.trainable_layers
                    ),
                    self.config.train_max_grad_norm,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            global_step += 1
            self.accelerator.log({"train_loss": train_loss}, step=global_step)
        return global_step

    def _config_check(self) -> Tuple[bool, str]:
        samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        total_train_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.train_gradient_accumulation_steps
        )

        if not self.config.sample_batch_size >= self.config.train_batch_size:
            return (
                False,
                f"Sample batch size ({self.config.sample_batch_size}) must be greater than or equal to the train batch size ({self.config.train_batch_size})",
            )
        if not self.config.sample_batch_size % self.config.train_batch_size == 0:
            return (
                False,
                f"Sample batch size ({self.config.sample_batch_size}) must be divisible by the train batch size ({self.config.train_batch_size})",
            )
        if not samples_per_epoch % total_train_batch_size == 0:
            return (
                False,
                f"Number of samples per epoch ({samples_per_epoch}) must be divisible by the total train batch size ({total_train_batch_size})",
            )
        return True, ""

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            if self.accelerator.is_main_process:
                log_with_time(f"Epoch {epoch} | Global Step: {global_step}")
            global_step = self.step(epoch, global_step)

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL DDPO Model") -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL DDPO Model`.
        """
        try:
            user = whoami()["name"]
        # handle the offline case
        except:  # noqa
            warnings.warn("Cannot retrieve user information assuming you are running in offline mode.")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}")
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card(save_directory)

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
import torch.nn.functional as F
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from transformers import BitsAndBytesConfig

from ..models import DDPOStableDiffusionPipeline, DDPOEmu1LMMPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
from pdb import set_trace as bp
from pynvml import *


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


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


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


def log_with_time(message):
    c = datetime.now()
    print(f"time: {c} | {message}")


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


class DDPOEmu1LMMTrainer(BaseTrainer):
    """
    The DDPOEmuTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models.
    Note, this trainer is heavily inspired by the work here: https://github.com/kvablack/ddpo-pytorch
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        **config** (`DDPOConfig`) -- Configuration object for DDPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **reward_function** (Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]) -- Reward function to be used
        **prompt_function** (Callable[[], Tuple[str, Any]]) -- Function to generate prompts to guide model
        **sd_pipeline** (`DDPOStableDiffusionPipeline`) -- Stable Diffusion pipeline to be used for training.
        **image_samples_hook** (Optional[Callable[[Any, Any, Any], Any]]) -- Hook to be called to log images
    """

    _tag_names = ["trl", "ddpo"]

    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DDPOEmu1LMMPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        self.config = config
        self.image_samples_callback = image_samples_hook

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

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps,
            **self.config.accelerator_kwargs,
        )

        is_okay, message = self._config_check()
        if not is_okay:
            raise ValueError(message)

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(ddpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
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

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.emu_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

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
        with torch.no_grad():
            self.neg_prompt_embed = self.sd_pipeline._get_negative_prompt_embedding("")

        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            emu_encoder = self.accelerator.prepare(trainable_layers)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, emu_encoder.parameters()))
        else:
            self.trainable_layers = self.accelerator.prepare(trainable_layers)

        if self.config.async_reward_computation:
            self.executor = futures.ThreadPoolExecutor(max_workers=config.max_workers)

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

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
                rewards.append(
                    (
                        torch.as_tensor(reward, device=self.accelerator.device),
                        reward_metadata,
                    )
                )
        else:
            rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
            rewards = [
                (torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result())
                for reward, reward_metadata in rewards
            ]

        return zip(*rewards)

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
        with torch.no_grad():
            samples, prompt_image_data, unet_conditions = self._generate_samples(
                iterations=self.config.sample_num_batches_per_epoch,
                batch_size=self.config.sample_batch_size,
            )

            # for key in samples[0].keys():
            #     for i in range(len(samples)):
            #         print(f"key: {key} | value: {samples[i][key].shape}")
            #     print("=============================")
            # exit()

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            # processing the prompts separately because they are not tensors
            samples_prompts = []
            for s in samples:
                samples_prompts.extend(s["prompts"])

            # processing the remaining keys
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys() if k != "prompts"}

            rewards, rewards_metadata = self.compute_rewards(
                prompt_image_data, is_async=self.config.async_reward_computation
            )

            for i, image_data in enumerate(prompt_image_data):
                image_data.extend([rewards[i], rewards_metadata[i]])

            if self.image_samples_callback is not None:
                if self.accelerator.is_main_process:
                    self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])

            rewards = torch.cat(rewards)
            rewards = self.accelerator.gather(rewards).cpu().numpy()

            self.accelerator.log(
                {
                    "reward": rewards,
                    "epoch": epoch,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                },
                step=global_step,
            )

            if self.config.per_prompt_stat_tracking:
                # gather the prompts across processes
                prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                advantages = self.stat_tracker.update(prompts, rewards)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # ungather advantages;  keep the entries corresponding to the samples on this process
            samples["advantages"] = (
                torch.as_tensor(advantages)
                .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
                .to(self.accelerator.device)
            )

        # del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        for inner_epoch in range(self.config.train_num_inner_epochs):
            with torch.no_grad():
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=self.accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}
                samples_prompts = [samples_prompts[i] for i in perm]

                # shuffle along time dimension independently for each sample
                # still trying to understand the code below
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
                )

                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                        perms,
                    ]

                original_keys = samples.keys()
                original_values = samples.values()
                # rebatch them as user defined train_batch_size is different from sample_batch_size
                reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

                # Transpose the list of original values
                transposed_values = zip(*reshaped_values)
                # Create new dictionaries for each row of transposed values
                samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]
                samples_prompts_batched = [
                    samples_prompts[i * self.config.train_batch_size : (i + 1) * self.config.train_batch_size]
                    for i in range(len(samples_prompts) // self.config.train_batch_size)
                ]

            self.sd_pipeline.emu_encoder.train()
            global_step = self._train_batched_samples(
                inner_epoch, epoch, global_step, samples_batched, samples_prompts_batched, unet_conditions
            )
            # ensure optimization step at the end of the inner epoch
            if not self.accelerator.sync_gradients:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )

        if epoch != 0 and epoch % self.config.save_freq == 0:
            self.accelerator.save_state()

        return global_step

    def calculate_ppo_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds, unet_conditions):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    encoder_hidden_states=embeds,
                    cross_attention_kwargs=unet_conditions[0],
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=embeds,
                    cross_attention_kwargs=unet_conditions[0],
                ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds, unet_conditions):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    encoder_hidden_states=embeds,
                    cross_attention_kwargs=unet_conditions[0],
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=embeds,
                    cross_attention_kwargs=unet_conditions[0],
                ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

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
    def _generate_samples(self, iterations, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (List[Dict[str, torch.Tensor]]), prompt_image_pairs (List[List[Any]])
        """
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.emu_encoder.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for s_idx in range(iterations):
            if self.accelerator.is_main_process:
                log_with_time(f"Generating samples: {s_idx}/{iterations}")
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])

            prompt_output = self.sd_pipeline.tokenizer(prompts, padding="max_length", return_tensors="pt")
            prompt_ids = prompt_output.input_ids.to(self.accelerator.device)
            attention_mask = prompt_output.attention_mask.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.emu_encoder.generate_image(prompts)

            with self.autocast():
                sd_output, unet_conditions = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)

            samples.append(
                {
                    "prompts": prompts,
                    "prompt_ids": prompt_ids,
                    "attention_mask": attention_mask,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs, unet_conditions

    def get_prompt_embeds(self, prompt_batch, negative_prompt_key=""):

        batch_size = len(prompt_batch)
        print_gpu_utilization()
        neg_prompt_embed = self.sd_pipeline.emu_encoder.generate_image([negative_prompt_key])
        print_gpu_utilization()
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
        print_gpu_utilization()
        sample_prompt_embeds = self.sd_pipeline.emu_encoder.generate_image(prompt_batch)
        print_gpu_utilization()
        exit()

        return sample_neg_prompt_embeds, sample_prompt_embeds

    def get_prompt_embeds_teacher_forcing(
        self, prompt_batch, gt_prompt_embeds, gt_negative_prompt_embeds, negative_prompt_key=""
    ):

        batch_size = len(prompt_batch)
        sample_prompt_embeds = self.sd_pipeline.emu_encoder.teacher_forcing(prompt_batch, gt_prompt_embeds)
        # sample_prompt_embeds = self.sd_pipeline.emu_encoder.generate_image(prompt_batch)
        neg_prompt_embed = self.sd_pipeline.emu_encoder.teacher_forcing(
            [negative_prompt_key], gt_negative_prompt_embeds
        )
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)

        return sample_neg_prompt_embeds, sample_prompt_embeds

    def _train_batched_samples(
        self, inner_epoch, epoch, global_step, batched_samples, samples_prompts_batched, unet_conditions
    ):
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
        torch.cuda.empty_cache()
        info = defaultdict(list)
        for i, sample in enumerate(batched_samples):
            if self.accelerator.is_main_process:
                log_with_time(f"Training batched samples: {i}/{len(batched_samples)}")

            # neg_prompt_embeds, prompt_embeds = self.get_prompt_embeds_teacher_forcing(
            #     samples_prompts_batched[i],
            #     batched_samples[i]["prompt_embeds"],
            #     batched_samples[i]["negative_prompt_embeds"],
            # )
            neg_prompt_embeds, prompt_embeds = self.get_prompt_embeds(
                samples_prompts_batched[i],
            )
            print("=" * 50)
            print(samples_prompts_batched[i])
            print("original", batched_samples[i]["prompt_embeds"].mean(-1))
            print("teacher forcing", prompt_embeds.mean(-1))
            print("original", batched_samples[i]["negative_prompt_embeds"].mean(-1))
            print("teacher forcing", neg_prompt_embeds.mean(-1))
            print("=" * 50)
            exit()

            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes

                # if self.accelerator.is_main_process:
                #     print("============================")
                #     print(f"neg_prompt_embeds")
                #     print(neg_prompt_embeds.shape)
                #     print(neg_prompt_embeds)
                #     print("-" * 50)
                #     print(f"neg_prompt_embeds, gt")
                #     print(sample["negative_prompt_embeds"].shape)
                #     print(sample["negative_prompt_embeds"])
                #     print("-" * 50)
                #     print(f"prompt_embeds")
                #     print(prompt_embeds.shape)
                #     print(prompt_embeds)
                #     print("-" * 50)
                #     print(f"prompt_embeds, gt")
                #     print(sample["prompt_embeds"].shape)
                #     print(sample["prompt_embeds"])
                #     print("============================")
                #     print_gpu_utilization()
                #     exit()
                embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
            else:
                embeds = prompt_embeds

            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.emu_encoder):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                        unet_conditions,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)
                    bp()

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
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
                print("success")
                exit()
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

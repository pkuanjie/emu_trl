from functools import partial
from typing import List, Optional

import torch

from torch import nn
from .causal_former import CausalFormer
from .model import MultimodalCfg, CLIPVisionCfg, VLadapterCfg, _build_vision_tower
from .transformer import LayerNorm
from .prediction_mixin import PredictClassMixin
from pdb import set_trace as bp

try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        StoppingCriteriaList,
        MaxLengthCriteria,
    )
except ImportError as e:
    pass

from transformers.generation.configuration_utils import GenerationConfig
from pynvml import *

GENERATION_CONFIG = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=32000)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


class Emu(nn.Module, PredictClassMixin):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        vladapter_cfg: VLadapterCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
        args=None,
        apply_lemmatizer=False,
        prompt=None,
    ):
        super().__init__()

        self.args = args

        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        vladapter_cfg = VLadapterCfg(**vladapter_cfg) if isinstance(vladapter_cfg, dict) else vladapter_cfg

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            cast_dtype=cast_dtype,
        )
        if vision_cfg.freeze:
            self.visual.requires_grad_(False)
            self.visual = self.visual.eval()

        norm_layer = partial(LayerNorm, eps=1e-6)

        self.ln_visual = norm_layer(vision_cfg.width)
        nn.init.constant_(self.ln_visual.bias, 0)
        nn.init.constant_(self.ln_visual.weight, 1.0)

        from .modeling_llama import LLaMAForClsAndRegression

        self.decoder = LLaMAForClsAndRegression(args=args)

        if multimodal_cfg.freeze:
            self.decoder.requires_grad_(False)
            self.decoder.eval()

        self.cformer = CausalFormer(
            args=args,
            n_causal=vladapter_cfg.n_causal,
            vision_width=vision_cfg.width,
            output_dim=self.decoder.config.d_model,
        )

        self.n_causal = vladapter_cfg.n_causal
        self.pad_id = pad_id

        self.prompt = prompt
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.image_placeholder = "[IMG]" + "<image>" * self.n_causal + "[/IMG]"

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.cformer.set_grad_checkpointing()
        self.decoder.set_grad_checkpointing()

    def forward(
        self, image, text_input, input_mask, text_output=None, output_mask=None, image_latent=None, image_features=None
    ):
        # [B, C, H, W] --> [B, n_patch, C_vis]
        if image_latent is None or image_features is None:
            image_features = self.visual.forward_features(image)
        # ln for visual features
        image_features = self.ln_visual(image_features)
        # [B, n_patch, C_vis] --> [B, n_causal, C_llm]
        image_features = self.cformer(image_features)
        # loss from hf lm model
        loss = self.decoder(
            image_features,
            text_input=text_input,
            text_output=text_output,
            text_mask=input_mask,
            output_mask=output_mask,
        )
        return loss

    # @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_new_tokens=50,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=0.0,
        num_captions=1,
        temperature=1,
        penalty_alpha=None,  # contrastive search
        top_k=None,
        no_repeat_ngram_size=None,
        **kwargs,
    ):
        GENERATION_CONFIG.pad_token_id = self.decoder.tokenizer.pad_token_id
        GENERATION_CONFIG.bos_token_id = self.decoder.tokenizer.bos_token_id
        GENERATION_CONFIG.eos_token_id = self.decoder.tokenizer.eos_token_id

        image = samples["image"]
        if image is not None:
            image = image.to(dtype=torch.bfloat16)
            image_features = self.ln_visual(self.visual.forward_features(image))
            image_features = self.cformer(image_features).squeeze().to(dtype=torch.bfloat16)

        prompt = samples["prompt"] if "prompt" in samples.keys() else self.prompt

        from .modeling_llama import LLaMAForClsAndRegression

        if isinstance(self.decoder, LLaMAForClsAndRegression):
            self.decoder.tokenizer.padding_side = "left"

        input_tokens = self.decoder.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.args.device)

        self.decoder.tokenizer.padding_side = "right"

        input_ids = input_tokens.input_ids
        encoder_atts = input_tokens.attention_mask

        img_token_id = self.decoder.tokenizer.convert_tokens_to_ids("<image>")  # 32003
        img_token_idx_list = input_ids.eq(img_token_id)

        with torch.amp.autocast(device_type=self.args.device.type, dtype=torch.bfloat16):
            if self.args.instruct:
                inputs_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.decoder.lm.model.embed_tokens(input_ids)

            if image is not None:
                image_features = image_features.reshape(-1, image_features.shape[-1])
                inputs_embeds[img_token_idx_list] = image_features

            inputs_embeds = inputs_embeds
            encoder_atts = encoder_atts

            outputs = self.decoder.lm.generate(
                generation_config=GENERATION_CONFIG,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs,
            )

            output_text = self.decoder.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    # @torch.no_grad()
    def teacher_forcing(
        self,
        text: List[str],
        target_image_embeds: torch.Tensor,
        max_token_length: int = 256,
        image: Optional[torch.Tensor] = None,
        placeholder: str = "[<IMG_PLH>]",
    ) -> torch.Tensor:
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids(["<image>", "[IMG]"])
        device = self.ln_visual.weight.device

        # print_gpu_utilization()
        if image is not None:
            # image placeholder is already injected into text prompt
            prompt_image_embeds = self.visual.forward_features(image)
            prompt_image_embeds = self.ln_visual(prompt_image_embeds)
            prompt_image_embeds = self.cformer(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])

        text = [t.replace(placeholder, self.image_placeholder) for t in text]
        # print("text", text)
        # print_gpu_utilization()

        for num_img_token in range(self.n_causal + 1):
            # print(num_img_token)
            # print_gpu_utilization()
            if num_img_token == 0:
                text = [f"{t}[IMG]" for t in text]
            else:
                text = [f"{t}<image>" for t in text]
        # print("text", text)

        # print_gpu_utilization()
        self.decoder.tokenizer.model_max_length = max_token_length
        inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt")
        # inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt")
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)
        # print("input_ids", input_ids.shape, input_ids)

        # print_gpu_utilization()
        text_embeds = self.decoder.lm.model.embed_tokens(input_ids)
        # print("text_embs", text_embeds.shape, text_embeds.mean(-1))

        image_idx = input_ids == IMAGE
        cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
        if image is not None:
            prompt_idx = torch.logical_and(image_idx, cumsum_idx > (self.n_causal))
            text_embeds[prompt_idx] = prompt_image_embeds

        # print_gpu_utilization()
        if target_image_embeds is not None:
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= (self.n_causal)))
            text_embeds[target_idx] = target_image_embeds
            # print("initial_target_idx", target_idx)
            # print("text_embs_with_img", text_embeds.shape, text_embeds.mean(-1))

        # print_gpu_utilization()

        # print(text_embeds.shape, attention_mask.shape)
        outputs = self.decoder.lm.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # print_gpu_utilization()
        image_idx = (input_ids == IMAGE) + (input_ids == BOI)
        cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
        target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= (self.n_causal + 1)))
        # print("final_target_idx", target_idx)

        # print_gpu_utilization()
        hidden_states = outputs.hidden_states[-1]
        target_image_embeds = hidden_states[target_idx]
        target_image_embeds = target_image_embeds[:-1, :]
        target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
        target_image_embeds = target_image_embeds.type(self.decoder.lm.stu_regress_head.weight.dtype)
        target_image_embeds = self.decoder.lm.stu_regress_head(target_image_embeds)
        # print("target_image_embeds", target_image_embeds.shape, target_image_embeds.mean(-1))
        # exit()

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds

    # @torch.no_grad()
    def generate_image_efficient(
        self,
        text: List[str],
        max_token_length: int = 64,
        image: Optional[torch.Tensor] = None,
        placeholder: str = "[<IMG_PLH>]",
    ) -> torch.Tensor:
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids(["<image>", "[IMG]"])
        device = self.ln_visual.weight.device

        # print_gpu_utilization()
        if image is not None:
            # image placeholder is already injected into text prompt
            prompt_image_embeds = self.visual.forward_features(image)
            prompt_image_embeds = self.ln_visual(prompt_image_embeds)
            prompt_image_embeds = self.cformer(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])

        text = [t.replace(placeholder, self.image_placeholder) for t in text]
        # print_gpu_utilization()

        # =====================================
        # move from loop out to here
        # print(num_img_token)
        # print_gpu_utilization()
        # if num_img_token == 0:
        text = [f"{t}[IMG]" for t in text]
        # else:
        #     text = [f"{t}<image>" for t in text]
        if image is not None:
            for _ in range(prompt_image_embeds.shape[0]):
                text = [f"{t}<image>" for t in text]

        # print_gpu_utilization()
        # if image is not None:
        #     self.decoder.tokenizer.model_max_length = 64
        # else:
        #     self.decoder.tokenizer.model_max_length = 32
        self.decoder.tokenizer.model_max_length = max_token_length
        self.decoder.tokenizer.padding_side = "left"
        inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt", truncation=True)
        # inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt")
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)

        # print_gpu_utilization()
        text_embeds = self.decoder.lm.model.embed_tokens(input_ids)
        # print("text_embs", text_embeds.shape, text_embeds.mean(-1))

        image_idx = input_ids == IMAGE
        cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
        if image is not None:
            prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
            text_embeds[prompt_idx] = prompt_image_embeds
        # =====================================

        target_image_embeds = None

        target_idx_before = torch.zeros_like(image_idx)
        target_idx_after = torch.zeros_like(image_idx)
        target_idx_after[:, -1] = 1

        for num_img_token in range(self.n_causal):

            # text_embeds shape: 1 x 32 x 5120
            # print(num_img_token)
            # print_gpu_utilization()
            if target_image_embeds is not None:
                text_embeds = torch.cat([text_embeds, torch.ones_like(text_embeds[:, 0:1, :])], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, 0:1])], dim=1)
                target_idx_before = torch.cat([target_idx_before, torch.ones_like(target_idx_before[:, 0:1])], dim=1)
                target_idx_after = torch.cat([target_idx_after, torch.ones_like(target_idx_after[:, 0:1])], dim=1)
                text_embeds[target_idx_before] = target_image_embeds
                # print("initial_target_idx", target_idx)
                # print("text_embs_with_img", text_embeds.shape, text_embeds.mean(-1))

            # print_gpu_utilization()

            # print(text_embeds.shape, attention_mask.shape)
            outputs = self.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

            # print_gpu_utilization()

            # print_gpu_utilization()
            hidden_states = outputs.last_hidden_state  # 1 x 32 x 5120
            # print("hidden_states", hidden_states.shape, hidden_states.mean(-1))
            target_image_embeds = hidden_states[target_idx_after]  # 1 x 5120
            # print("target_image_embeds", target_image_embeds.shape, target_image_embeds.mean(-1))
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = target_image_embeds.type(self.decoder.lm.stu_regress_head.weight.dtype)
            target_image_embeds = self.decoder.lm.stu_regress_head(target_image_embeds)
            # print("target_image_embeds", target_image_embeds.shape, target_image_embeds.mean(-1))
            # print_gpu_utilization()
            del outputs
            if num_img_token < self.n_causal - 1:
                del hidden_states
            torch.cuda.empty_cache()

        # print(target_image_embeds.shape)
        # print_gpu_utilization()
        _, C = target_image_embeds.shape
        B = target_idx_before.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds

    # @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        max_token_length: int = 64,
        image: Optional[torch.Tensor] = None,
        placeholder: str = "[<IMG_PLH>]",
    ) -> torch.Tensor:
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids(["<image>", "[IMG]"])
        device = self.ln_visual.weight.device

        # print_gpu_utilization()
        if image is not None:
            # image placeholder is already injected into text prompt
            prompt_image_embeds = self.visual.forward_features(image)
            prompt_image_embeds = self.ln_visual(prompt_image_embeds)
            prompt_image_embeds = self.cformer(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])

        text = [t.replace(placeholder, self.image_placeholder) for t in text]
        # print_gpu_utilization()

        target_image_embeds = None
        for num_img_token in range(self.n_causal):
            # print(num_img_token)
            # print_gpu_utilization()
            if num_img_token == 0:
                text = [f"{t}[IMG]" for t in text]
            else:
                text = [f"{t}<image>" for t in text]

            # print_gpu_utilization()
            self.decoder.tokenizer.model_max_length = max_token_length
            inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt")
            # inputs = self.decoder.tokenizer(text, padding="max_length", return_tensors="pt")
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device)

            # print_gpu_utilization()
            text_embeds = self.decoder.lm.model.embed_tokens(input_ids)
            # print("text_embs", text_embeds.shape, text_embeds.mean(-1))

            image_idx = input_ids == IMAGE
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds

            # print_gpu_utilization()
            if target_image_embeds is not None:
                target_idx = torch.logical_and(
                    image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token)
                )
                text_embeds[target_idx] = target_image_embeds
                # print("initial_target_idx", target_idx)
                # print("text_embs_with_img", text_embeds.shape, text_embeds.mean(-1))

            # print_gpu_utilization()

            # print(text_embeds.shape, attention_mask.shape)
            outputs = self.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

            # print_gpu_utilization()
            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(
                image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token + 1)
            )
            # print("final_target_idx", target_idx)

            # print_gpu_utilization()
            hidden_states = outputs.last_hidden_state
            # print("hidden_states", hidden_states.shape, hidden_states.mean(-1))
            target_image_embeds = hidden_states[target_idx]
            # print("target_image_embeds", target_image_embeds.shape, target_image_embeds.mean(-1))
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = target_image_embeds.type(self.decoder.lm.stu_regress_head.weight.dtype)
            target_image_embeds = self.decoder.lm.stu_regress_head(target_image_embeds)
            # print("target_image_embeds", target_image_embeds.shape, target_image_embeds.mean(-1))
            # print_gpu_utilization()
            del outputs
            if num_img_token < self.n_causal - 1:
                del hidden_states
            torch.cuda.empty_cache()

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)
        # print_gpu_utilization()

        return target_image_embeds

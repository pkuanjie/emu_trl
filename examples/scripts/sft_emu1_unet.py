# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
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
"""
python examples/scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import tempfile
import torch.nn as nn
import wandb
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser
from torchvision import transforms

from trl import (
    DDPOConfig,
    SFTConfig,
    DDPOEmu1Trainer,
    DefaultDDPOStableDiffusionPipeline,
    DefaultDDPOEmu1Pipeline,
    SFTEmu1Trainer,
)
from trl.import_utils import is_npu_available, is_xpu_available
from trl.datasets.data_loader import get_loader
import wandb
import datetime

wandb.login()


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="./pretrain/emu1_ckpts/models--BAAI--Emu/snapshots/9d5face1ae9d8f5cd5c0ed891dc09e47833d06e1/pretrain",
        metadata={"help": "the pretrained model to use"},
    )
    main_name: str = field(
        default="sft_emu1_unet",
        metadata={"help": "the name of all models and dirs"},
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    resolution: int = field(default=512, metadata={"help": "the resolution to use for training"})
    random_flip: bool = field(default=True, metadata={"help": "the resolution to use for training"})
    center_crop: bool = field(default=False, metadata={"help": "the resolution to use for training"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # # For the sake of this example, we will only log the last batch of images
    # # and associated data
    # result = {}
    # images, prompts, _, rewards, _ = image_data[-1]

    # for i, image in enumerate(images):
    #     prompt = prompts[i]
    #     reward = rewards[i].item()
    #     result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0)

    # accelerate_logger.log_images(
    #     result,
    #     step=global_step,
    # )

    # this is a hack to force wandb to log the images as JPEGs instead of PNGs
    images, prompts, _, rewards, _ = image_data[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, image in enumerate(images):
            pil = Image.fromarray((image.cpu().to(torch.float32).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil = pil.resize((256, 256))
            pil.save(os.path.join(tmpdir, f"{i}.jpg"))
        accelerate_logger.log(
            {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{i}.jpg"),
                        caption=f"{prompt:.25} | {reward:.2f}",
                    )
                    for i, (prompt, reward) in enumerate(zip(prompts, rewards))  # only log rewards from process 0
                ],
            },
            step=global_step,
        )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    args, sft_config = parser.parse_args_into_dataclasses()
    time_str = "{date:%Y_%m_%d_%H_%M}".format(date=datetime.datetime.now())
    sft_config.project_kwargs = {
        "logging_dir": f"./logs/logs_{args.main_name}_{time_str}",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"./save/save_{args.main_name}_{time_str}",
    }

    # remove the project directory if it exists so that it will not cause issues
    # os.system(f"rm -rf {ddpo_config.project_kwargs['project_dir']}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_loader = get_loader(
        image_processor=train_transforms,
        text_processor=None,
        batch_size=sft_config.train_batch_size,
        num_workers=1,
        shuffle=True,
        args=None,
        data_dirs="./data/ze",
        drop_last=True,
    )

    test_prompts = [
        "A tranquil lake, framed by majestic mountains, reflects the warm hues of the setting sun. Tall pine trees line the shore, their emerald needles swaying gently in the evening breeze. Wildflowers dot the landscape, painting it with splashes of vibrant color. The air is filled with the soothing symphony of chirping birds and buzzing insects, creating a serene ambiance that captivates the soul.",
        "In a sun-drenched meadow, playful puppies chase each other in circles, their tails wagging with unbridled joy. Nearby, a curious kitten pounces on fallen leaves, its tiny claws batting at the rustling foliage. A family of deer gracefully leaps through the grass, their coats shimmering in the golden light. Overhead, a kaleidoscope of colorful birds flit and flutter, adding to the lively scene.",
        "The bustling streets of a vibrant city pulse with energy as people go about their daily lives. Pedestrians navigate crowded sidewalks, weaving through a sea of faces with purposeful strides. Street vendors beckon passersby with tantalizing aromas of sizzling street food, while musicians serenade the crowd with lively tunes. From open-air markets to bustling cafes, every corner is alive with activity and excitement, creating an unforgettable tapestry of urban existence.",
        "Standing atop a windswept hill, a lone figure gazes up at the star-studded sky with wonder and awe. Each twinkling star seems to whisper secrets of the universe, filling the night with a sense of mystery and possibility. In this moment of solitude, the person feels a profound connection to the cosmos, as if they are part of something much larger than themselves. It's a reminder of the boundless beauty and infinite potential that surrounds us.",
    ]

    pipeline = DefaultDDPOEmu1Pipeline(args.pretrained_model, use_lora=args.use_lora)

    trainer = SFTEmu1Trainer(
        sft_config,
        train_loader,
        test_prompts,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # test dataloader
    # for i, data in enumerate(train_loader):
    #     print(i)
    #     print(data)
    #     if i == 3:
    #         break

    # trainer.push_to_hub(args.hf_hub_model_id)

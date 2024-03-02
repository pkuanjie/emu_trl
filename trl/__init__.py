# flake8: noqa

__version__ = "0.7.11.dev0"

from .core import set_seed
from .environment import TextEnvironment, TextHistory
from .extras import BestOfNSampler
from .import_utils import (
    is_bitsandbytes_available,
    is_diffusers_available,
    is_npu_available,
    is_peft_available,
    is_wandb_available,
    is_xpu_available,
)
from .models import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PreTrainedModelWrapper,
    create_reference_model,
    setup_chat_format,
)
from .trainer import (
    DataCollatorForCompletionOnlyLM,
    DPOTrainer,
    IterativeSFTTrainer,
    PPOConfig,
    PPOTrainer,
    RewardConfig,
    RewardTrainer,
    SFTTrainer,
)


if is_diffusers_available():
    from .models import (
        DDPOPipelineOutput,
        DDPOSchedulerOutput,
        DDPOStableDiffusionPipeline,
        DefaultDDPOStableDiffusionPipeline,
        DDPOEmu1Pipeline,
        DDPOEmu1LMMPipeline,
        DDPOEmu1LMMUNetPipeline,
        DDPOEmu2Pipeline,
        DefaultDDPOEmu1Pipeline,
        DefaultDDPOEmu1LMMPipeline,
        DefaultDDPOEmu1LMMUNetPipeline,
        DefaultDDPOEmu2Pipeline,
    )
    from .trainer import (
        DDPOConfig,
        SFTConfig,
        DDPOTrainer,
        DDPOEmu1Trainer,
        SFTEmu1Trainer,
        DDPOEmu1LMMTrainer,
        DDPOEmu1LMMUNetTrainer,
        DDPOEmu2Trainer,
    )

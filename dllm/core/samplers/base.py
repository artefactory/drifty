from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class BaseSamplerOutput:
    sequences: torch.Tensor
    histories: list[torch.Tensor] | None = None


@dataclass
class BaseSamplerOutputCompleteHistory:
    sequences: torch.Tensor | None = None 
    histories_x: list[torch.Tensor] | None = None
    histories_x0: list[torch.Tensor] | None = None
    histories_logprobs: list[torch.Tensor] | None = None
    histories_unmask_logprobs: list[torch.Tensor] | None = None
    histories_mask: list[torch.Tensor] | None = None
    histories_accepted: list[torch.Tensor] | None = None
    histories_remasking: list[torch.Tensor] | None = None
    histories_H: list[torch.Tensor] | None = None
    histories_entropy: list[torch.Tensor] | None = None
    histories_semantic_entropy: list[torch.Tensor] | None = None
    histories_semantic_dispersion: list[torch.Tensor] | None = None
    histories_semantic_logprobs: list[torch.Tensor] | None = None
    histories_semantic_bestlogprobs_label: list[torch.Tensor] | None = None
    histories_num_transfer_tokens: list[torch.Tensor] | None = None
    sample_indices: torch.Tensor | None = None
    step_per_block: int | None = None
    block_size: int | None = None
    start_idx_history: list[int] | None = None
    max_new_tokens: int| None = None
    attention_mask: torch.Tensor| None = None


    


@dataclass
class BaseSamplerConfig:
    return_dict: bool = False


@dataclass
class BaseSampler(ABC):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    scheduler: BaseAlphaScheduler | None = None

    def __post_init__(self):
        if self.scheduler is None:
            self.scheduler = LinearAlphaScheduler()

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        prompts: list[torch.Tensor | list],
        config: BaseSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: BaseSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError

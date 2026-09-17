"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput, BaseSamplerOutputCompleteHistory
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens, get_num_transfer_tokens_remasking


def compute_semantic_dispersion(
    p: torch.Tensor,
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute semantic dispersion D = 1 - ||sum_i p_i * e_hat_i||^2
    where e_hat_i are L2-normalized embedding vectors.

    High D = model hesitates between semantically distant tokens.
    Low D  = model hesitates between semantically close tokens (or is confident).

    Args:
        p: Probability tensor of shape [B, T, V].
        embedding_matrix: L2-normalized embedding matrix of shape [V, D].

    Returns:
        Tensor of shape [B, T] with dispersion values in [0, 1].
    """
    # p: [B, T, V], embedding_matrix: [V, D]
    # weighted_emb = sum_i p_i * e_hat_i  -> [B, T, D]
    weighted_emb = torch.matmul(p, embedding_matrix.to(p.dtype))  # [B, T, D]
    # D = 1 - ||weighted_emb||^2
    dispersion = 1.0 - (weighted_emb * weighted_emb).sum(dim=-1)  # [B, T]
    return dispersion.clamp(min=0.0)  # numerical safety





def computeSemanticMetrics(
    p: torch.Tensor,
    save_path_token_to_cluster: str = "PipelineTest/embeddings/token_to_cluster_stc_3k.json",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate token probabilities into semantic cluster probabilities and return
    the semantic entropy per batch item and sequence position.

    Args:
        p: Probability tensor of shape [B, T, V].
        save_path_token_to_cluster: JSON path containing token_id -> cluster_id.

    Returns:
        Tuple of (p_labels, semantic_entropy, bestLabels) where:
        - p_labels: Tensor of shape [B, T, num_labels] containing semantic cluster probabilities.
        - semantic_entropy: Tensor of shape [B, T] containing semantic entropy values.
        - bestLabels: Tensor of shape [B, T] containing the most probable semantic cluster labels.
    """
    if p.ndim != 3:
        raise ValueError(f"p must have shape [B, T, V], got {tuple(p.shape)}")

    with open(save_path_token_to_cluster, "r") as f:
        tokens_to_cluster = json.load(f)

    batch_size, seq_len, vocab_size = p.shape
    cluster_ids = torch.full((vocab_size,), -1, dtype=torch.long, device=p.device)

    for token_id_str, cluster_id in tokens_to_cluster.items():
        token_id = int(token_id_str)
        cluster_id = int(cluster_id)
        if 0 <= token_id < vocab_size:
            cluster_ids[token_id] = cluster_id

    valid_mask = cluster_ids >= 0
    if not torch.any(valid_mask):
        return torch.zeros((batch_size, seq_len, 0), dtype=p.dtype, device=p.device), torch.zeros((batch_size, seq_len), dtype=p.dtype, device=p.device), torch.zeros((batch_size, seq_len), dtype=torch.long, device=p.device)

    num_labels = int(cluster_ids[valid_mask].max().item()) + 1
    p_labels = torch.zeros(
        (batch_size, seq_len, num_labels), dtype=p.dtype, device=p.device
    )

    # Only gather from valid token indices to avoid allocating a full [B, T, V] mask tensor
    valid_indices = torch.where(valid_mask)[0]  # [num_valid]
    p_valid = p[:, :, valid_indices]  # [B, T, num_valid]
    cluster_ids_valid = cluster_ids[valid_indices].view(1, 1, -1).expand(batch_size, seq_len, -1)
    p_labels.scatter_add_(dim=-1, index=cluster_ids_valid, src=p_valid)
    del p_valid, cluster_ids_valid

    p_labels = p_labels + 1e-14
    best_pLabels = torch.gather(p_labels, dim=-1, index=torch.argmax(p_labels, dim=-1).unsqueeze(-1)).squeeze(-1)
    entropy = -(p_labels * torch.log(p_labels + 1e-10)).sum(dim=-1)
    bestLabels = torch.argmax(p_labels, dim=-1)

    return best_pLabels, entropy, bestLabels


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    additional_steps: int = 0  # Only used for the remasking variant with complete history, adds extra steps after all masks are revealed


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking in {"min_entropy", "entropy"}:
                    p = F.softmax(logits, dim=-1)
                    x0_p = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -x0_p  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking in {"min_entropy", "entropy"}:
                    p = F.softmax(logits, dim=-1)
                    x0_p = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -x0_p  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
        








@dataclass
class MDLMSamplerWithCompleteHistory(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutputCompleteHistory | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget

        histories_x = [] if return_dict else None
        histories_x0 = [] if return_dict else None
        histories_logprobs = [] if return_dict else None
        histories_unmask_logprobs = [] if return_dict else None
        histories_mask = [] if return_dict else None
        histories_entropy = [] if return_dict else None
        histories_H = [] if return_dict else None
        histories_semantic_entropy = [] if return_dict else None
        histories_semantic_dispersion = [] if return_dict else None
        histories_semantic_logprobs = [] if return_dict else None
        histories_semantic_bestlogprobs_label = [] if return_dict else None


        # Precompute L2-normalized embedding matrix for semantic dispersion
        if return_dict:
            emb_matrix = self.model.get_input_embeddings().weight.detach().float()  # [V, D]
            emb_matrix = F.normalize(emb_matrix, p=2, dim=-1)
        else:
            emb_matrix = None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                if histories_mask is not None:
                    histories_mask.append(mask_index.clone())
                
                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                


                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if histories_x0 is not None:
                    histories_x0.append(x0.clone())

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                p = F.softmax(logits, dim=-1)
                if remasking == "low_confidence":
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking in {"min_entropy", "entropy"}:
                    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -entropy  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region

                if histories_logprobs is not None:
                    histories_logprobs.append(x0_p.clone()) 

                if histories_semantic_entropy is not None:
                    p_labels, semantic_entropy, bestLabels = computeSemanticMetrics(p)
                    histories_semantic_entropy.append(semantic_entropy.clone())
                    histories_semantic_logprobs.append(p_labels.clone())
                    histories_semantic_bestlogprobs_label.append(bestLabels.clone())
                    del p_labels, semantic_entropy, bestLabels

                if histories_entropy is not None:
                    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [B, T]
                    histories_entropy.append(entropy.clone())
                    del entropy

                if histories_semantic_dispersion is not None:
                    dispersion = compute_semantic_dispersion(p, emb_matrix)
                    histories_semantic_dispersion.append(dispersion.clone())
                    del dispersion

                target_ids = x[~mask_index] 
                unmask_logits_distirbution = p[~mask_index]
                del p  # free [B, T, V] tensor
                torch.cuda.empty_cache()
                specific_logits = torch.gather(unmask_logits_distirbution, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                del unmask_logits_distirbution
                full_logprobs_vector = torch.zeros_like(x, dtype=specific_logits.dtype)
                full_logprobs_vector[~mask_index] = specific_logits

                if histories_unmask_logprobs is not None:
                    histories_unmask_logprobs.append(full_logprobs_vector.clone())

                H = x0_p - full_logprobs_vector  # [B, T]: p_max - p_token_unmasked
                if histories_H is not None:
                    histories_H.append(H.clone())

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories_x is not None:
                    histories_x.append(x.clone())


        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutputCompleteHistory(sequences=x, 
                                                    histories_x=histories_x, 
                                                    histories_x0 = histories_x0, 
                                                    histories_logprobs = histories_logprobs, 
                                                    histories_unmask_logprobs = histories_unmask_logprobs, 
                                                    histories_mask=histories_mask, 
                                                    histories_entropy=histories_entropy, 
                                                    histories_H=histories_H, 
                                                    histories_semantic_dispersion=histories_semantic_dispersion, 
                                                    histories_semantic_entropy=histories_semantic_entropy, 
                                                    histories_semantic_logprobs=histories_semantic_logprobs, 
                                                    histories_semantic_bestlogprobs_label=histories_semantic_bestlogprobs_label, 
                                                    step_per_block = steps, 
                                                    block_size=block_size, 
                                                    start_idx_history= prompt_lens, 
                                                    max_new_tokens=max_new_tokens, 
                                                    attention_mask= attention_mask)
    
    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutputCompleteHistory | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories_x = [] if return_dict else None
        histories_x0 = [] if return_dict else None
        histories_logprobs = [] if return_dict else None
        histories_mask = [] if return_dict else None
        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id
                if histories_mask is not None:
                    histories_mask.append(mask_index_full)

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                if histories_logprobs is not None:
                    histories_logprobs.append(logits_with_noise.clone())

                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if histories_x0 is not None:
                    histories_x0.append(x0.clone())

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking in {"min_entropy", "entropy"}:
                    p = F.softmax(logits, dim=-1)
                    x0_p = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -x0_p  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories_x is not None:
                    histories_x.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutputCompleteHistory(sequences=x, histories_x=histories_x, histories_x0=histories_x0, histories_logprobs=histories_logprobs, histories_mask=histories_mask, step_per_block=steps, block_size = block_size)
        



@dataclass
class MDLMSamplerRemaskingWithCompleteHistory(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutputCompleteHistory | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        additional_steps = kwargs.get("additional_steps", config.additional_steps)

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)  # per-block step budget
        additional_steps_per_block = math.ceil(additional_steps / num_blocks) if additional_steps > 0 else 0
        
        histories_x = [] if return_dict else None
        histories_x0 = [] if return_dict else None
        histories_logprobs = [] if return_dict else None
        histories_unmask_logprobs = [] if return_dict else None
        histories_mask = [] if return_dict else None
        histories_remasking = [] if return_dict else None
        histories_H = [] if return_dict else None
        histories_num_transfer_tokens = [] if return_dict else None

        num_transfer_tokens = get_num_transfer_tokens_remasking(
            x, mask_id, num_blocks, steps, additional_steps, scheduler=self.scheduler
        )

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            

            # Some steps may be skipped if there are no transfers
            effective_steps = steps_per_block + additional_steps_per_block
            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                                
                mask_index = x == mask_id  # current global mask map

                

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                


                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if histories_x0 is not None:
                    histories_x0.append(x0.clone())

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                p = F.softmax(logits, dim=-1)
                if remasking == "low_confidence":
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking in {"min_entropy", "entropy"}:
                    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -entropy  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region

                if histories_logprobs is not None:
                    histories_logprobs.append(x0_p.clone()) 

                target_ids = x[~mask_index] 
                unmask_logits_distirbution = p[~mask_index]
                specific_logits = torch.gather(unmask_logits_distirbution, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                full_logprobs_vector = torch.zeros_like(x, dtype=specific_logits.dtype)
                full_logprobs_vector[~mask_index] = specific_logits

                if histories_unmask_logprobs is not None:
                    histories_unmask_logprobs.append(full_logprobs_vector.clone())
                
                H = x0_p - full_logprobs_vector  # [B, T]: p_max - p_token_unmasked
                if histories_H is not None:
                    histories_H.append(H.clone())
                
                # Skip remasking on the last step to avoid stranded masks
                if i < effective_steps - 1:
                    # Remasking avec probabilité H_i selon une loi de Bernoulli
                    # Seules les positions démasquées et avec H > 0 peuvent être remasquées
                    remasking_mask_step = torch.zeros(
                        (B, T), dtype=torch.bool, device=x.device
                    )
                    for j in range(B):
                        generation_region = torch.zeros(
                            T, dtype=torch.bool, device=x.device
                        )
                        generation_start = prompt_lens[j]
                        generation_end = min(prompt_lens[j] + max_new_tokens, T)
                        if generation_start < generation_end:
                            generation_region[generation_start:generation_end] = True

                        remask_mask = (~mask_index[j]) & (H[j] > 0) & generation_region
                        if remask_mask.any():
                            remask_positions = torch.where(remask_mask)[0]
                            remask_probs = torch.clamp(H[j][remask_positions], 0, 1)
                            remask_samples = torch.bernoulli(remask_probs)
                            remask_idx = remask_positions[remask_samples.bool()]

                            nb_remaining_steps = num_transfer_tokens.shape[1] - (b *(steps_per_block+additional_steps_per_block) + i + 1)
                            nb_remasked = int(remask_samples.sum().item())
                            
                            
                            if nb_remaining_steps > 0 and nb_remasked > 0:
                                base = nb_remasked // nb_remaining_steps
                                remainder = nb_remasked % nb_remaining_steps
                                for future_step in range(b *(steps_per_block+additional_steps_per_block) + i + 1, num_transfer_tokens.shape[1]):
                                    num_transfer_tokens[j, future_step] += base
                                if remainder > 0:
                                    future_steps = torch.arange(b*(steps_per_block+additional_steps_per_block) +
                                        i + 1,
                                        num_transfer_tokens.shape[1],
                                        device=num_transfer_tokens.device,
                                        dtype=torch.long,
                                    )
                                    future_vals = num_transfer_tokens[j, future_steps]
                                    _, order = torch.sort(future_vals)
                                    target_steps = future_steps[order[:remainder]]
                                    num_transfer_tokens[j, target_steps] += 1

                            remasking_mask_step[j, remask_idx] = True
                            x[j, remask_idx] = mask_id

                            
                    if histories_num_transfer_tokens is not None:
                        histories_num_transfer_tokens.append(num_transfer_tokens.clone())

                    if histories_remasking is not None:
                        histories_remasking.append(remasking_mask_step.clone())
                else:
                    if histories_remasking is not None:
                        remasking_mask_step = torch.zeros(
                            (B, T), dtype=torch.bool, device=x.device
                        )
                        histories_remasking.append(remasking_mask_step.clone())

                mask_index = x == mask_id

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, b *(steps_per_block+additional_steps_per_block) + i].item()
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories_x is not None:
                    histories_x.append(x.clone())
                    
                mask_index = x == mask_id  # current global mask map

                if histories_mask is not None:
                    histories_mask.append(mask_index)

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutputCompleteHistory(sequences=x, 
                                                    histories_x=histories_x, 
                                                    histories_x0 = histories_x0, 
                                                    histories_logprobs = histories_logprobs, 
                                                    histories_unmask_logprobs = histories_unmask_logprobs, 
                                                    histories_mask=histories_mask, 
                                                    histories_remasking=histories_remasking, 
                                                    histories_H=histories_H, 
                                                    histories_num_transfer_tokens=histories_num_transfer_tokens, 
                                                    step_per_block = steps, 
                                                    block_size=block_size, 
                                                    start_idx_history= prompt_lens, 
                                                    max_new_tokens=max_new_tokens, 
                                                    attention_mask= attention_mask)
    

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutputCompleteHistory | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories_x = [x.clone()] if return_dict else None
        histories_x0 = [] if return_dict else None
        histories_logprobs = [] if return_dict else None
        histories_mask = [] if return_dict else None

        num_transfer_tokens = torch.zeros((B, steps + addition), dtype=torch.long, device=self.model.device)  
        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id
                if histories_mask is not None:
                    histories_mask.append(mask_index_full)

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                if histories_logprobs is not None:
                    histories_logprobs.append(logits_with_noise.clone())

                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if histories_x0 is not None:
                    histories_x0.append(x0.clone())

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking in {"min_entropy", "entropy"}:
                    p = F.softmax(logits, dim=-1)
                    x0_p = -(p * torch.log(p + 1e-8)).sum(dim=-1)
                    x0_p = -x0_p  # maximize negative entropy <=> select minimum entropy
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories_x is not None:
                    histories_x.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutputCompleteHistory(sequences=x, histories_x=histories_x, histories_x0=histories_x0, histories_logprobs=histories_logprobs, histories_mask=histories_mask, step_per_block=steps, block_size = block_size)
        
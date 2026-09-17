"""
DiffusionGemma sampler for the dllm sampler interface.

Example:
    python -u examples/diffusiongemma/sample.py \
        --model_name_or_path /mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it
"""

import math
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput, BaseSamplerOutputCompleteHistory


@dataclass
class DiffusionGemmaSamplerOutput(BaseSamplerOutput):
    tokens_per_forward: torch.Tensor | None = None
    decoder_forward_passes: torch.Tensor | None = None


@dataclass
class DiffusionGemmaSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    steps: int = 48
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    entropy_bound: float = 0.1
    temperature: float | None = None
    max_temperature: float = 0.8
    min_temperature: float = 0.4
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@dataclass
class DiffusionGemmaSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list] | torch.Tensor,
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> DiffusionGemmaSamplerOutput | torch.Tensor:
        if config is None:
            config = DiffusionGemmaSamplerConfig()
        return_dict = bool(kwargs.pop("return_dict", config.return_dict))

        model_config = getattr(self.model, "config", None)
        text_config = model_config.text_config
        canvas_length = model_config.canvas_length
        encoder = self.model.model.encoder
        decoder = self.model.model.decoder
        decoder_dtype = decoder.embed_tokens.weight.dtype
        device = next(self.model.parameters()).device
        if device.type == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_token_id = config.pad_token_id
        if pad_token_id is None and self.tokenizer.pad_token_id is not None:
            pad_token_id = int(self.tokenizer.pad_token_id)
        eos_token_id = config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(model_config, "eos_token_id", None)

        if isinstance(inputs, torch.Tensor):
            prompts = [inputs] if inputs.dim() == 1 else [row for row in inputs]
        elif inputs and all(isinstance(token_id, int) for token_id in inputs):
            prompts = [inputs]
        else:
            prompts = list(inputs)
        if not prompts:
            raise ValueError("DiffusionGemmaSampler requires at least one input prompt.")

        tensors = [
            prompt.to(device=device, dtype=torch.long)
            if isinstance(prompt, torch.Tensor)
            else torch.as_tensor(prompt, dtype=torch.long, device=device)
            for prompt in prompts
        ]
        max_prompt_len = max(tensor.numel() for tensor in tensors)
        input_ids = torch.full(
            (len(tensors), max_prompt_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            # Left-pad: right-align each prompt so the last real prompt token
            # sits at the same column for every row in the batch. This is
            # required so that generated (canvas) tokens are always adjacent
            # (RoPE distance 1) to each sample's real last prompt token,
            # regardless of how much shorter it is than the batch's longest
            # prompt. Right-padding here would inflate that distance for
            # shorter prompts and corrupt the start of their generation.
            n = tensor.numel()
            input_ids[i, max_prompt_len - n:] = tensor.reshape(-1)
            attention_mask[i, max_prompt_len - n:] = True

        batch_size = input_ids.shape[0]
        max_new_tokens = config.max_new_tokens
        attention_mask = kwargs.pop("attention_mask", attention_mask).to(device=device)

        if len(attention_mask.shape) > 2:
            raise ValueError("`attention_mask` passed to `sample` must be 2D.")
        attention_mask = attention_mask.bool()

        cur_len = input_ids.shape[1]
        initial_input_ids_len = cur_len
        max_new_canvases = math.ceil(max_new_tokens / canvas_length)

        device = input_ids.device
        eos_tensor = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        decoder_forward_passes = torch.zeros(batch_size, dtype=torch.int, device=device)
        if eos_token_id is not None:
            eos_tensor = torch.tensor(eos_token_id, device=input_ids.device)

        cache_kwargs = {}
        if hasattr(model_config, "get_text_config"):
            cache_kwargs["config"] = model_config
        past_key_values = DynamicCache(**cache_kwargs)

        # Per-row real position ids: left-padded columns get position 0
        # (irrelevant, masked out by attention_mask); real prompt tokens get
        # their true 0-indexed position regardless of how much left padding
        # precedes them.
        encoder_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0).to(torch.int32)
        prompt_lens_tensor = attention_mask.sum(dim=-1).to(torch.int32)
        decoder_position_ids = prompt_lens_tensor.unsqueeze(1) + torch.arange(
            canvas_length, dtype=torch.int32, device=device,
        ).unsqueeze(0)

        entropy_bound = float(config.entropy_bound)

        vocab_size = int(text_config.vocab_size)
        confidence_threshold = config.entropy_threshold

        decoder_attention_mask = torch.nn.functional.pad(
            attention_mask, (0, canvas_length), value=True
        )

        is_prefill = True
        for _ in range(max_new_canvases):
            encoder_input_ids = input_ids if is_prefill else input_ids[:, -canvas_length:]
            encoder_input_ids = encoder_input_ids.clone(memory_format=torch.contiguous_format)
            create_encoder_masks = getattr(encoder, "create_masks_for_generate", None)
            if create_encoder_masks is None:
                encoder_attention_mask = attention_mask
            else:
                dummy_input_embeds = torch.empty(
                    (encoder_input_ids.shape[0], encoder_input_ids.shape[1], 0),
                    dtype=text_config.dtype,
                    device=encoder_input_ids.device,
                )
                encoder_attention_mask = create_encoder_masks(
                    config=model_config,
                    inputs_embeds=dummy_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=encoder_position_ids,
                )
            encoder_outputs = encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                position_ids=encoder_position_ids,
            )
            past_key_values = encoder_outputs.past_key_values
            is_prefill = False

            current_canvas = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, canvas_length),
                device=device,
            )
            self_conditioning_logits = None

            create_decoder_mask = getattr(decoder, "create_diffusion_decoder_attention_mask", None)
            mask_mapping = (
                decoder_attention_mask
                if create_decoder_mask is None
                else create_decoder_mask(
                    config=text_config,
                    inputs_embeds=current_canvas.unsqueeze(-1),
                    past_key_values=past_key_values,
                    decoder_attention_mask=decoder_attention_mask,
                )
            )
            finished_denoising = torch.zeros(batch_size, dtype=torch.bool, device=device)
            argmax_canvas_history = None
            argmax_canvas = current_canvas

            for cur_step in reversed(range(1, config.steps + 1)):
                decoder_forward_passes += ~(finished_denoising | finished_sequences)

                decoder_outputs = self.model.forward(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=past_key_values,
                    decoder_position_ids=decoder_position_ids,
                )
                raw_logits = decoder_outputs.logits
                cur_step_tensor = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
                processed_logits = raw_logits
                if config.temperature:
                    processed_logits = processed_logits / config.temperature
                elif config.min_temperature is not None and config.max_temperature is not None:
                    temperature = config.min_temperature + (
                        (config.max_temperature - config.min_temperature)
                        * (cur_step_tensor / config.steps)
                    )
                    processed_logits = processed_logits / temperature
                probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)

                denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
                denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
                new_argmax_canvas = torch.argmax(processed_logits, dim=-1)

                dist = torch.distributions.Categorical(logits=processed_logits)
                token_entropy = dist.entropy()
                sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
                cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
                sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
                accepted_token_mask = torch.scatter(
                    input=torch.zeros_like(sorted_selection_mask),
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_selection_mask,
                )
                accepted_canvas = torch.where(accepted_token_mask, denoiser_canvas, current_canvas).clone()
                random_canvas = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=accepted_canvas.shape,
                    device=accepted_canvas.device,
                )
                new_current_canvas = torch.where(~accepted_token_mask, random_canvas, accepted_canvas).clone()

                if config.stability_threshold is not None and confidence_threshold is not None:
                    (
                        new_argmax_canvas,
                        new_current_canvas,
                        processed_logits,
                        finished_denoising,
                        argmax_canvas_history,
                    ) = self._apply_adaptive_stopping(
                        finished_denoising=finished_denoising,
                        current_canvas=current_canvas,
                        new_current_canvas=new_current_canvas,
                        argmax_canvas=argmax_canvas,
                        new_argmax_canvas=new_argmax_canvas,
                        processed_logits=processed_logits,
                        self_conditioning_logits=self_conditioning_logits,
                        argmax_canvas_history=argmax_canvas_history,
                        token_entropy=token_entropy,
                        stability_threshold=config.stability_threshold,
                        confidence_threshold=confidence_threshold,
                    )

                current_canvas = new_current_canvas
                argmax_canvas = new_argmax_canvas
                self_conditioning_logits = processed_logits.to(decoder_dtype)

                if torch.all(finished_denoising):
                    break

            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
            finished_this_canvas = torch.zeros_like(finished_sequences)
            if eos_tensor is not None:
                finished_this_canvas |= torch.isin(input_ids[:, -canvas_length:], eos_tensor).any(dim=-1)

            previously_finished_sequences = finished_sequences
            finished_sequences = previously_finished_sequences | finished_this_canvas
            if pad_token_id is not None and torch.any(finished_sequences):
                input_ids[previously_finished_sequences, -canvas_length:] = pad_token_id
                if eos_tensor is not None and torch.any(finished_this_canvas):
                    new_tokens = input_ids[:, -canvas_length:]
                    is_eos = torch.isin(new_tokens, eos_tensor)
                    eos_cumsum = is_eos.cumsum(dim=-1)
                    pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
                    new_tokens[pad_mask] = pad_token_id

            if torch.all(finished_sequences):
                break

            cur_len += canvas_length
            attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)
            decoder_attention_mask = torch.nn.functional.pad(decoder_attention_mask, (0, canvas_length), value=True)
            # Positions are per-row (rows differ in real prompt length), so
            # shift each row's own previous canvas positions forward instead
            # of rebuilding a single arange shared across the batch.
            encoder_position_ids = decoder_position_ids
            decoder_position_ids = encoder_position_ids + canvas_length

        new_tokens = input_ids[:, initial_input_ids_len:]
        if pad_token_id is not None:
            num_valid_tokens = (new_tokens != pad_token_id).sum(dim=-1)
        else:
            num_valid_tokens = torch.full(
                (input_ids.shape[0],),
                new_tokens.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        tokens_per_forward = num_valid_tokens / decoder_forward_passes.clamp_min(1)
        output = DiffusionGemmaSamplerOutput(
            sequences=input_ids,
            histories=None,
            tokens_per_forward=tokens_per_forward,
            decoder_forward_passes=decoder_forward_passes,
        )
        return output if return_dict else input_ids

    @staticmethod
    def _apply_adaptive_stopping(
        *,
        finished_denoising: torch.Tensor,
        current_canvas: torch.Tensor,
        new_current_canvas: torch.Tensor,
        argmax_canvas: torch.Tensor,
        new_argmax_canvas: torch.Tensor,
        processed_logits: torch.Tensor,
        self_conditioning_logits: torch.Tensor | None,
        argmax_canvas_history: torch.Tensor | None,
        token_entropy: torch.Tensor,
        stability_threshold: int,
        confidence_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if finished_denoising.any():
            new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
            new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
            processed_logits = torch.where(
                finished_denoising[:, None, None],
                self_conditioning_logits,
                processed_logits,
            )

        if stability_threshold == 0:
            stable = torch.ones(
                (processed_logits.shape[0]),
                device=processed_logits.device,
                dtype=torch.bool,
            )
        else:
            if argmax_canvas_history is None:
                argmax_canvas_history = torch.full(
                    (
                        stability_threshold,
                        new_argmax_canvas.shape[0],
                        new_argmax_canvas.shape[1],
                    ),
                    -1,
                    dtype=new_argmax_canvas.dtype,
                    device=new_argmax_canvas.device,
                )
            stable = (argmax_canvas_history == new_argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            argmax_canvas_history = torch.roll(argmax_canvas_history, shifts=-1, dims=0)
            argmax_canvas_history[-1] = new_argmax_canvas

        if finished_denoising.any():
            confident = torch.mean(torch.distributions.Categorical(logits=processed_logits).entropy(), dim=-1)
        else:
            confident = torch.mean(token_entropy, dim=-1)
        finished_denoising |= stable & (confident < confidence_threshold)

        return (
            new_argmax_canvas,
            new_current_canvas,
            processed_logits,
            finished_denoising,
            argmax_canvas_history,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> DiffusionGemmaSamplerOutput | torch.Tensor:
        del inputs, config, kwargs
        raise NotImplementedError("DiffusionGemma native generator does not expose infill in dllm yet.")


@dataclass
class DiffusionGemmaSamplerWithCompleteHistory(BaseSampler):
    """DiffusionGemma sampler that records full trajectory histories.

    Returns BaseSamplerOutputCompleteHistory compatible with GetInfoFromBaseSamplerOutput.
    The "mask" here is ~accepted_token_mask (True = token still uncertain/being denoised).
    """

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list] | torch.Tensor,
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutputCompleteHistory | torch.Tensor:
        if config is None:
            config = DiffusionGemmaSamplerConfig()
        return_dict = bool(kwargs.pop("return_dict", config.return_dict))

        model_config = getattr(self.model, "config", None)
        text_config = model_config.text_config
        canvas_length = getattr(config, "canvas_length", None) or model_config.canvas_length
        encoder = self.model.model.encoder
        decoder = self.model.model.decoder
        decoder_dtype = decoder.embed_tokens.weight.dtype
        device = next(self.model.parameters()).device
        if device.type == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pad_token_id = config.pad_token_id
        if pad_token_id is None and self.tokenizer.pad_token_id is not None:
            pad_token_id = int(self.tokenizer.pad_token_id)
        eos_token_id = config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(model_config, "eos_token_id", None)

        if isinstance(inputs, torch.Tensor):
            prompts = [inputs] if inputs.dim() == 1 else [row for row in inputs]
        elif inputs and all(isinstance(token_id, int) for token_id in inputs):
            prompts = [inputs]
        else:
            prompts = list(inputs)
        if not prompts:
            raise ValueError("DiffusionGemmaSamplerWithCompleteHistory requires at least one input prompt.")

        tensors = [
            prompt.to(device=device, dtype=torch.long)
            if isinstance(prompt, torch.Tensor)
            else torch.as_tensor(prompt, dtype=torch.long, device=device)
            for prompt in prompts
        ]
        max_prompt_len = max(tensor.numel() for tensor in tensors)
        input_ids = torch.full(
            (len(tensors), max_prompt_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            # Left-pad: right-align each prompt so the last real prompt token
            # sits at the same column for every row in the batch (see
            # DiffusionGemmaSampler.sample for the full rationale).
            n = tensor.numel()
            input_ids[i, max_prompt_len - n:] = tensor.reshape(-1)
            attention_mask[i, max_prompt_len - n:] = True

        batch_size = input_ids.shape[0]
        max_new_tokens = config.max_new_tokens
        attention_mask = kwargs.pop("attention_mask", attention_mask).to(device=device)
        attention_mask = attention_mask.bool()

        cur_len = input_ids.shape[1]
        initial_input_ids_len = cur_len
        max_new_canvases = math.ceil(max_new_tokens / canvas_length)

        eos_tensor = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            eos_tensor = torch.tensor(eos_token_id, device=device)

        cache_kwargs = {}
        if hasattr(model_config, "get_text_config"):
            cache_kwargs["config"] = model_config
        past_key_values = DynamicCache(**cache_kwargs)

        # Per-row real position ids (see DiffusionGemmaSampler.sample): left
        # padding means the absolute column of "generation start" is the same
        # (initial_input_ids_len) for every row, but each row's true prompt
        # length differs, so RoPE positions must be computed per row.
        encoder_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0).to(torch.int32)
        prompt_lens_tensor = attention_mask.sum(dim=-1).to(torch.int32)
        decoder_position_ids = prompt_lens_tensor.unsqueeze(1) + torch.arange(
            canvas_length, dtype=torch.int32, device=device,
        ).unsqueeze(0)

        entropy_bound = float(config.entropy_bound)
        vocab_size = int(text_config.vocab_size)
        confidence_threshold = config.entropy_threshold

        decoder_attention_mask = torch.nn.functional.pad(
            attention_mask, (0, canvas_length), value=True
        )

        # History tracking (aligned to full sequence = prompt + generated tokens)
        histories_x = []
        histories_x0 = []
        histories_logprobs = []
        histories_entropy = []
        histories_mask = []
        histories_accepted = []  # raw accepted_token_mask per step (non-monotone)

        is_prefill = True
        for canvas_idx in range(max_new_canvases):
            encoder_input_ids = input_ids if is_prefill else input_ids[:, -canvas_length:]
            encoder_input_ids = encoder_input_ids.clone(memory_format=torch.contiguous_format)
            create_encoder_masks = getattr(encoder, "create_masks_for_generate", None)
            if create_encoder_masks is None:
                encoder_attention_mask = attention_mask
            else:
                dummy_input_embeds = torch.empty(
                    (encoder_input_ids.shape[0], encoder_input_ids.shape[1], 0),
                    dtype=text_config.dtype, device=encoder_input_ids.device,
                )
                encoder_attention_mask = create_encoder_masks(
                    config=model_config, inputs_embeds=dummy_input_embeds,
                    attention_mask=attention_mask, past_key_values=past_key_values,
                    position_ids=encoder_position_ids,
                )
            encoder_outputs = encoder(
                input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                past_key_values=past_key_values, position_ids=encoder_position_ids,
            )
            past_key_values = encoder_outputs.past_key_values
            is_prefill = False

            current_canvas = torch.randint(0, vocab_size, (batch_size, canvas_length), device=device)
            self_conditioning_logits = None

            create_decoder_mask = getattr(decoder, "create_diffusion_decoder_attention_mask", None)
            mask_mapping = (
                decoder_attention_mask
                if create_decoder_mask is None
                else create_decoder_mask(
                    config=text_config, inputs_embeds=current_canvas.unsqueeze(-1),
                    past_key_values=past_key_values, decoder_attention_mask=decoder_attention_mask,
                )
            )
            argmax_canvas = current_canvas

            # Track which positions in this canvas have been "accepted" cumulatively
            ever_accepted = torch.zeros(batch_size, canvas_length, dtype=torch.bool, device=device)

            # Compute how many tokens to unmask per step (like MDLM scheduler)
            total_to_unmask = canvas_length
            num_transfer_per_step = []
            remaining = total_to_unmask
            for s in range(config.steps):
                n = remaining // (config.steps - s)
                num_transfer_per_step.append(n)
                remaining -= n

            print(f"Num transfer per step: {num_transfer_per_step}")

            for step_idx in range(config.steps):
                decoder_outputs = self.model.forward(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=past_key_values,
                    decoder_position_ids=decoder_position_ids,
                )
                raw_logits = decoder_outputs.logits
                processed_logits = raw_logits
                if config.temperature:
                    processed_logits = processed_logits / config.temperature
                elif config.min_temperature is not None and config.max_temperature is not None:
                    progress = (config.steps - step_idx) / config.steps
                    temperature = config.min_temperature + (
                        (config.max_temperature - config.min_temperature) * progress
                    )
                    processed_logits = processed_logits / temperature
                probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
                new_argmax_canvas = torch.argmax(processed_logits, dim=-1)

                # Entropy and confidence per token
                token_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=new_argmax_canvas.unsqueeze(-1)).squeeze(-1)

                # Select top-k most confident tokens among non-accepted positions
                k = num_transfer_per_step[step_idx]
                confidence = torch.where(~ever_accepted, x0_p, torch.tensor(-float('inf'), device=device))
                if k > 0:
                    _, topk_indices = torch.topk(confidence, k=k, dim=-1)
                    accepted_token_mask = torch.zeros(batch_size, canvas_length, dtype=torch.bool, device=device)
                    accepted_token_mask.scatter_(1, topk_indices, True)
                else:
                    accepted_token_mask = torch.zeros(batch_size, canvas_length, dtype=torch.bool, device=device)

                # Update cumulative acceptance (monotone)
                ever_accepted = ever_accepted | accepted_token_mask

                # Update canvas: accepted positions get argmax token, others re-randomized
                current_canvas = torch.where(
                    ever_accepted, new_argmax_canvas,
                    torch.randint(0, vocab_size, (batch_size, canvas_length), device=device),
                )
                argmax_canvas = new_argmax_canvas
                self_conditioning_logits = processed_logits.to(decoder_dtype)

                # Record histories
                full_len = initial_input_ids_len + (canvas_idx + 1) * canvas_length
                mask_full = torch.zeros(batch_size, full_len, dtype=torch.bool, device=device)
                mask_full[:, initial_input_ids_len + canvas_idx * canvas_length:] = ~ever_accepted
                histories_mask.append(mask_full)

                accepted_full = torch.zeros(batch_size, full_len, dtype=torch.bool, device=device)
                accepted_full[:, initial_input_ids_len + canvas_idx * canvas_length:
                              initial_input_ids_len + (canvas_idx + 1) * canvas_length] = accepted_token_mask
                histories_accepted.append(accepted_full)

                x0_full = input_ids.clone()
                x0_full = torch.nn.functional.pad(x0_full, (0, full_len - x0_full.shape[1]), value=pad_token_id or 0)
                x0_full[:, initial_input_ids_len + canvas_idx * canvas_length:
                         initial_input_ids_len + (canvas_idx + 1) * canvas_length] = new_argmax_canvas
                histories_x0.append(x0_full)

                logprobs_full = torch.zeros(batch_size, full_len, device=device)
                logprobs_full[:, initial_input_ids_len + canvas_idx * canvas_length:
                              initial_input_ids_len + (canvas_idx + 1) * canvas_length] = x0_p
                histories_logprobs.append(logprobs_full)

                entropy_full = torch.zeros(batch_size, full_len, device=device)
                entropy_full[:, initial_input_ids_len + canvas_idx * canvas_length:
                             initial_input_ids_len + (canvas_idx + 1) * canvas_length] = token_entropy
                histories_entropy.append(entropy_full)

                x_full = input_ids.clone()
                x_full = torch.nn.functional.pad(x_full, (0, full_len - x_full.shape[1]), value=pad_token_id or 0)
                x_full[:, initial_input_ids_len + canvas_idx * canvas_length:
                        initial_input_ids_len + (canvas_idx + 1) * canvas_length] = torch.where(
                    ever_accepted, new_argmax_canvas,
                    torch.zeros_like(new_argmax_canvas),
                )
                histories_x.append(x_full)

            # Commit best canvas to sequence
            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
            finished_this_canvas = torch.zeros_like(finished_sequences)
            if eos_tensor is not None:
                finished_this_canvas |= torch.isin(input_ids[:, -canvas_length:], eos_tensor).any(dim=-1)

            previously_finished_sequences = finished_sequences
            finished_sequences = previously_finished_sequences | finished_this_canvas
            if pad_token_id is not None and torch.any(finished_sequences):
                input_ids[previously_finished_sequences, -canvas_length:] = pad_token_id
                if eos_tensor is not None and torch.any(finished_this_canvas):
                    new_tokens = input_ids[:, -canvas_length:]
                    is_eos = torch.isin(new_tokens, eos_tensor)
                    eos_cumsum = is_eos.cumsum(dim=-1)
                    pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
                    new_tokens[pad_mask] = pad_token_id

            if torch.all(finished_sequences):
                break

            cur_len += canvas_length
            attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)
            decoder_attention_mask = torch.nn.functional.pad(decoder_attention_mask, (0, canvas_length), value=True)
            # Per-row positions: shift each row's own previous canvas
            # positions forward rather than rebuilding a batch-shared arange.
            encoder_position_ids = decoder_position_ids
            decoder_position_ids = encoder_position_ids + canvas_length

        if not return_dict:
            return input_ids

        output = BaseSamplerOutputCompleteHistory(
            sequences=input_ids,
            histories_x=histories_x,
            histories_x0=histories_x0,
            histories_logprobs=histories_logprobs,
            histories_entropy=histories_entropy,
            histories_mask=histories_mask,
            histories_accepted= histories_accepted,
            step_per_block=config.steps,
            block_size=canvas_length,
            # With left-padding, generation always starts at the same
            # absolute column (initial_input_ids_len) for every row in the
            # batch, regardless of each sample's real prompt length -- unlike
            # `prompt_lens`, which was wrong for any batch with mixed-length
            # prompts (history tensors are indexed by absolute column, see
            # the `full_len`/slicing above and dllm/pipelines/dream/sampler.py
            # for the analogous constant-offset convention).
            start_idx_history=[initial_input_ids_len] * batch_size,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
        )
        return output

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        del inputs, config, kwargs
        raise NotImplementedError
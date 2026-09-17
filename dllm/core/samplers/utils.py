import torch

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
    additional_steps: int = 0,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    result = torch.stack(padded_rows, dim=0)
    
    # Concatenate additional zero columns if requested
    if additional_steps > 0:
        additional_zeros = torch.zeros(
            result.size(0), additional_steps, dtype=result.dtype, device=result.device
        )
        result = torch.cat([result, additional_zeros], dim=1)
    
    return result

# def get_num_transfer_tokens_remasking(x, mask_id, num_blocks, block_size, steps, additional_steps=0, scheduler=None):
#     mask_index = x == mask_id
#     steps_per_block = steps // num_blocks
#     additional_steps_per_block = additional_steps // num_blocks
#     num_transfer_tokens = torch.zeros((x.size(0), steps + additional_steps), dtype=torch.long, device=x.device)
#     for i in range(num_blocks):
#         print('step per block:', steps_per_block, 'additional steps per block:', additional_steps_per_block)
#         num_transfer__token_in_block = get_num_transfer_tokens(
#             mask_index=mask_index[:, i * block_size:(i + 1) * block_size],
#             steps=steps_per_block,
#             scheduler=scheduler,
#             stochastic=False,
#         )
#         begin_index = i *(steps_per_block + additional_steps_per_block)
#         end_index = begin_index + steps_per_block

#         print(num_transfer_tokens[:, begin_index:end_index].shape, num_transfer__token_in_block.shape)
#         num_transfer_tokens[:, begin_index:end_index] += num_transfer__token_in_block # on veut uniquement conservé des 0 sur les additional steps  la fin de chaque block
#     return num_transfer_tokens

def get_num_transfer_tokens_remasking(x, mask_id, num_blocks, steps, additional_steps=0, scheduler=None):
    mask_index = x == mask_id
    steps_per_block = steps // num_blocks
    additional_steps_per_block = additional_steps // num_blocks
    num_transfer_tokens_with_additional_steps = torch.zeros((x.size(0), steps + additional_steps), dtype=torch.long, device=x.device)
    num_transfer_tokens = get_num_transfer_tokens(
        mask_index=mask_index,
        steps=steps,
        scheduler=scheduler,
        stochastic=False,
    )
    for i in range(num_blocks):
        begin_index = i *(steps_per_block + additional_steps_per_block)
        end_index = begin_index + steps_per_block
        num_transfer_tokens_with_additional_steps[:, begin_index:end_index] = num_transfer_tokens[:, i * steps_per_block:(i + 1) * steps_per_block]
    return num_transfer_tokens_with_additional_steps    

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise





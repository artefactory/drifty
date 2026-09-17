import dllm
import torch
from Levenshtein import distance


def _pad_slice(e: torch.Tensor, i: int, start_idx: int, length: int, pad_value=0):
    """Return e[i, start_idx:start_idx+length], right-padding e with pad_value
    first if it is narrower than start_idx+length.

    Samplers that generate in growing blocks (e.g. DiffusionGemma's canvases)
    record history tensors whose sequence length grows step over step, so a
    single histories_* list can mix tensors of different widths. A plain
    slice would silently return a shorter-than-expected array for the early
    steps instead of raising, corrupting anything that assumes a fixed
    max_new_tokens width.
    """
    end_idx = start_idx + length
    row = e[i]
    if row.shape[-1] < end_idx:
        row = torch.nn.functional.pad(row, (0, end_idx - row.shape[-1]), value=pad_value)
    return row[start_idx:end_idx]


def getLogProbs(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory) -> list[torch.Tensor]:
    nb_examples = outputs.histories_logprobs[0].shape[0]
    res_logProbs =[]
    for i in range(nb_examples):
        res_logProbs.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_logprobs])
    return res_logProbs


def getUnmaskLogProbs(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory) -> list[torch.Tensor]:
    nb_examples = outputs.histories_unmask_logprobs[0].shape[0]
    res_unmaskLogProbs =[]
    for i in range(nb_examples):
        res_unmaskLogProbs.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_unmask_logprobs])
    return res_unmaskLogProbs



def getEachStepGeneratedSequence(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory, tokenizer) -> list[list[str]]:
    nb_examples = outputs.histories_x[0].shape[0]
    res = []
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    for i in range(nb_examples):
        start_idx = outputs.start_idx_history[i]
        example_history_tokens = []
        for e in outputs.histories_x:
            token_ids = _pad_slice(e, i, start_idx, outputs.max_new_tokens, pad_value=pad_id)
            tokens = [tokenizer.decode(t) for t in token_ids]
            example_history_tokens.append(tokens)

        res.append(example_history_tokens)

    return res

def getEachStepProposedSequence(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory, tokenizer):
    nb_examples = outputs.histories_x0[0].shape[0]
    res = []
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    for i in range(nb_examples):
        start_idx = outputs.start_idx_history[i]
        example_history_tokens = []
        for e in outputs.histories_x0:
            token_ids = _pad_slice(e, i, start_idx, outputs.max_new_tokens, pad_value=pad_id)
            tokens = [tokenizer.decode(t) for t in token_ids]
            example_history_tokens.append(tokens)

        res.append(example_history_tokens)

    return res

def getEachStepProposedTokenIdSequence(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory, tokenizer):
    nb_examples = outputs.histories_x0[0].shape[0]
    res = []
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    for i in range(nb_examples):
        start_idx = outputs.start_idx_history[i]
        example_history_tokens = []
        for e in outputs.histories_x0:
            token_ids = _pad_slice(e, i, start_idx, outputs.max_new_tokens, pad_value=pad_id)
            example_history_tokens.append(token_ids.detach().cpu().numpy())

        res.append(example_history_tokens)

    return res


def getEachStepMask(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory, remask = False) -> list[torch.Tensor]:
    nb_examples = outputs.histories_mask[0].shape[0]
    m = outputs.histories_remasking if remask else outputs.histories_mask

    # DiffusionGemma's sampler records histories_mask *after* committing each
    # step's accepted tokens (dllm/pipelines/diffusiongemma/sampler.py), while
    # llada/dream record it *before* (dllm/core/samplers/mdlm.py) — entropy is
    # already pre-commit in both, only the mask is off by one for Gemma. Shift
    # it by one step so mask[t] means "still masked going into step t" for
    # every sampler family. histories_accepted is only ever populated by the
    # Gemma sampler, so it doubles as the family marker. The synthetic first
    # entry has zero width in the generated region, so _pad_slice's
    # pad_value=1 fills it entirely masked, matching the true pre-step-0 state.
    if not remask and outputs.histories_accepted:
        placeholder = m[0][:, :0]
        m = [placeholder] + m[:-1]

    res_masks =[]
    for i in range(nb_examples):
        # pad_value=1: positions not yet generated (beyond a step's recorded
        # width) are still masked/unresolved, same as a normal masked token.
        res_masks.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens, pad_value=1).detach().float().cpu().numpy() for e in m])
    return res_masks



def getEachStepChange(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    # Samplers like DiffusionGemma generate in growing blocks ("canvases"), so
    # tensors in histories_x0 don't all share the same sequence length: later
    # steps cover more committed tokens than earlier ones. Pad on the right to
    # the max length before stacking.
    max_len = max(t.shape[-1] for t in outputs.histories_x0)
    padded_x0 = [
        torch.nn.functional.pad(t, (0, max_len - t.shape[-1]))
        for t in outputs.histories_x0
    ]
    stacked_all = torch.stack(padded_x0)
    diff_mask = (stacked_all[1:] != stacked_all[:-1])
    diff_mask = diff_mask.permute(1, 0, 2)
    res_changes = []
    for i in range(diff_mask.shape[0]):
        start_idx = outputs.start_idx_history[i]

        example_diff = diff_mask[i, :, start_idx:start_idx+outputs.max_new_tokens]
        zero_row = torch.zeros(
            (1, example_diff.shape[1]), dtype=example_diff.dtype, device=example_diff.device
        )
        example_diff = torch.cat([zero_row, example_diff], dim=0)

        res_changes.append(example_diff.detach().cpu().numpy())
        
    return res_changes


from Levenshtein import distance
import numpy as np

def getLevenshtein(Proposed_sequences):
    # Proposed_sequences structure: [Batch, Steps, 128 tokens]
    res_levenshtein = []
    
    for example_history in Proposed_sequences:
        # example_history: [Steps, 128 tokens]
        nb_steps = len(example_history)
        example_changes = []
        
        for k in range(1, nb_steps):
            # On compare chaque mot à l'index 'm' entre l'étape k et k-1
            # On obtient une liste de 128 distances pour cette transition
            step_distances = [
                distance(example_history[k-1][m], example_history[k][m])
                for m in range(len(example_history[k]))
            ]
            example_changes.append(step_distances)
            
        # On convertit en array numpy pour faciliter le plot (Shape: [Steps-1, 128])
        res_levenshtein.append(np.array(example_changes))
        
    return res_levenshtein

def getH(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_H[0].shape[0]
    res_H =[]
    for i in range(nb_examples):
        res_H.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_H])
    return res_H


def getNumTransferTokens(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_num_transfer_tokens[0].shape[0]
    res_num_transfer_tokens =[]
    for i in range(nb_examples):
        res_num_transfer_tokens.append([e[i].detach().float().cpu().numpy() for e in outputs.histories_num_transfer_tokens]) 
    return res_num_transfer_tokens



def getEntropy(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_entropy[0].shape[0]
    res_entropy =[]
    for i in range(nb_examples):
        res_entropy.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_entropy])
    return res_entropy


def getSemanticDispersion(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_semantic_dispersion[0].shape[0]
    res = []
    for i in range(nb_examples):
        res.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_semantic_dispersion])
    return res


def getSemanticEntropy(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_semantic_entropy[0].shape[0]
    res = []
    for i in range(nb_examples):
        res.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_semantic_entropy])
    return res

def getSemanticLogprobs(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_semantic_logprobs[0].shape[0]
    res = []
    for i in range(nb_examples):
        res.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_semantic_logprobs])
    return res

def getSemanticBestLogprobsLabel(outputs:dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_semantic_bestlogprobs_label[0].shape[0]
    res = []
    for i in range(nb_examples):
        res.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().cpu().numpy() for e in outputs.histories_semantic_bestlogprobs_label])
    return res


def getEntropyJustUnmasked(outputs: dllm.core.samplers.BaseSamplerOutputCompleteHistory):
    nb_examples = outputs.histories_entropy[0].shape[0]
    res_entropy = []
    res_masks = []
    for i in range(nb_examples):
        res_entropy.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_entropy])
        res_masks.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens, pad_value=1).detach().float().cpu().numpy() for e in outputs.histories_mask])

    entropy_at_unmask_time_list = []
    for i in range(nb_examples):
        masked = np.asarray(res_masks[i], dtype=bool)
        ent = np.asarray(res_entropy[i], dtype=float)
        
        # 1. Calcul standard des transitions d'un pas à l'autre
        just_unmasked = np.zeros_like(masked, dtype=bool)
        just_unmasked[1:, :] = masked[:-1, :] & (~masked[1:, :])

        # 2. PRISE EN COMPTE DU DERNIER PAS DE DIFFUSION
        just_unmasked[-1, :] = just_unmasked[-1, :] | masked[-1, :]

        # =========================================================================
        # CORRECTION : Extraction à plat de TOUS les tokens validés de la trajectoire
        # =========================================================================
        # ent[just_unmasked] renvoie directement un tableau 1D contenant les entropies
        # de vos 32 tokens, sans distinction d'étape. Les étapes "vides" n'injectent aucun NaN.
        all_unmasked_tokens = ent[just_unmasked]
        entropy_at_unmask_time_list.append(all_unmasked_tokens)
        
    return entropy_at_unmask_time_list



def getLogProbsJustUnmasked(outputs):
    nb_examples = outputs.histories_logprobs[0].shape[0]
    res_logprobs = []
    res_masks = []
    for i in range(nb_examples):
        res_logprobs.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens).detach().float().cpu().numpy() for e in outputs.histories_logprobs])
        res_masks.append([_pad_slice(e, i, outputs.start_idx_history[i], outputs.max_new_tokens, pad_value=1).detach().float().cpu().numpy() for e in outputs.histories_mask])

    logprobs_at_unmask_time_list = []
    for i in range(nb_examples):
        masked = np.asarray(res_masks[i], dtype=bool)
        logp = np.asarray(res_logprobs[i], dtype=float)
        
        # 1. Calcul standard des transitions d'un pas à l'autre
        just_unmasked = np.zeros_like(masked, dtype=bool)
        just_unmasked[1:, :] = masked[:-1, :] & (~masked[1:, :])

        # 2. PRISE EN COMPTE DU DERNIER PAS DE DIFFUSION
        just_unmasked[-1, :] = just_unmasked[-1, :] | masked[-1, :]

        # =========================================================================
        # CORRECTION : Extraction à plat de TOUS les tokens validés de la trajectoire
        # =========================================================================
        # logp[just_unmasked] renvoie directement un tableau 1D contenant les log-probabilités
        # de vos 32 tokens, sans distinction d'étape. Les étapes "vides" n'injectent aucun NaN.
        all_unmasked_tokens = logp[just_unmasked]
        logprobs_at_unmask_time_list.append(all_unmasked_tokens)
        
    return logprobs_at_unmask_time_list
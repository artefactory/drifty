
import sys
import os
import re

import numpy as np

from GenerateBaseSamplerOutputsAndExtractInfo.GetInfoFromBaseSamplerOutput import getEntropy, getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getUnmaskLogProbs
import torch 
from dataclasses import fields
from dllm.core.samplers.base import BaseSamplerOutputCompleteHistory
from GenerateBaseSamplerOutputsAndExtractInfo.PlotResults import PlotlyEntropy, plotLogProbs, plotMasks, plotChanges,  PlotlyLogProbs,  getTxt, plotEntropy





def _pad_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(tensor.shape) == target_shape:
        return tensor

    padded = tensor.new_zeros(target_shape)
    slices = tuple(slice(0, s) for s in tensor.shape)
    padded[slices] = tensor
    return padded


def _pad_and_cat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]

    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        if t.dim() != len(max_shape):
            raise RuntimeError("Cannot merge tensors with different ranks")
        for dim_idx, dim_size in enumerate(t.shape):
            if dim_idx == 0:
                continue
            max_shape[dim_idx] = max(max_shape[dim_idx], dim_size)

    padded_tensors = []
    for t in tensors:
        target_shape = tuple([t.shape[0]] + max_shape[1:])
        padded_tensors.append(_pad_to_shape(t, target_shape))

    return torch.cat(padded_tensors, dim=0)

def mergeOutputsList(ouputpath='PipelineTest/results/outputs_no_remasking.pt'):
    outputs_list = torch.load(ouputpath, map_location="cpu", weights_only=False)

    if isinstance(outputs_list, BaseSamplerOutputCompleteHistory):
        return outputs_list
    if not isinstance(outputs_list, list):
        raise TypeError(f"Expected a list of BaseSampler outputs, got {type(outputs_list)}")
    if len(outputs_list) == 0:
        raise ValueError("outputs_list is empty")

    first = outputs_list[0]
    # Handle list of tuples (int, BaseSamplerOutputCompleteHistory, int)
    if isinstance(first, tuple):
        outputs_list = [item for item in outputs_list if isinstance(item, tuple)]
        outputs_list = [item[1] for item in outputs_list]
        first = outputs_list[0]
    if not isinstance(first, BaseSamplerOutputCompleteHistory):
        raise TypeError(
            "Expected list items of type BaseSamplerOutputCompleteHistory, "
            f"got {type(first)}"
        )

    merged = BaseSamplerOutputCompleteHistory()
    for f in fields(BaseSamplerOutputCompleteHistory):
        name = f.name
        values = [getattr(o, name) for o in outputs_list]
        non_none_values = [v for v in values if v is not None]

        if len(non_none_values) == 0:
            setattr(merged, name, None)
            continue

        sample = non_none_values[0]

        if isinstance(sample, torch.Tensor):
            setattr(merged, name, _pad_and_cat_tensors(non_none_values))
            continue

        if isinstance(sample, list):
            if len(sample) == 0:
                setattr(merged, name, [])
                continue

            if all(isinstance(v, list) for v in non_none_values):
                # Histories are lists of tensors per step; merge step-by-step across batches.
                if all(len(v) == len(non_none_values[0]) for v in non_none_values):
                    if all(
                        len(v) > 0 and isinstance(v[0], torch.Tensor)
                        for v in non_none_values
                    ):
                        merged_history = []
                        for step_idx in range(len(non_none_values[0])):
                            merged_history.append(
                                _pad_and_cat_tensors([v[step_idx] for v in non_none_values])
                            )
                        setattr(merged, name, merged_history)
                    else:
                        flat = []
                        for v in non_none_values:
                            flat.extend(v)
                        setattr(merged, name, flat)
                else:
                    flat = []
                    for v in non_none_values:
                        flat.extend(v)
                    setattr(merged, name, flat)
                continue

        # Scalar metadata (block_size, max_new_tokens, step_per_block): keep first non-None.
        setattr(merged, name, sample)

    return merged

def getEntropicCost(masks, changes, entropies):
    entropic_cost = []
    for j in range(len(masks)):
        changes_before_unmask = np.array(changes[j]) * (np.array(masks[j]))
        cost = ((changes_before_unmask) * np.array(entropies[j])).cumsum(axis=0) # cum sum
        entropic_cost.append(cost)
    return entropic_cost






def GetInfoFromIndex(listIndex, tokenizer, ouputpath='PipelineTest/results/outputs_no_remasking.pt'):

    outputs = mergeOutputsList(ouputpath)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    remasks = getEachStepMask(outputs, remask=True) if outputs.histories_remasking is not None else None
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    entropies = getEntropy(outputs) 
    if outputs.attention_mask is None:
        raise RuntimeError("Merged outputs has no attention_mask")
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    H = getH(outputs) if outputs.histories_H is not None else None
    entropic_costs = getEntropicCost(masks, changes, entropies)

    hmap = {}
    for i in range(len(outputs.sample_indices)):
        j = outputs.sample_indices[i].item() if outputs.sample_indices is not None else i
        hmap[j] = i
    
    l_index = [hmap[i] for i in listIndex]


    block_size = outputs.block_size 

    logprobs = [logprobs[i] for i in l_index]
    masks = [masks[i] for i in l_index]
    if remasks is not None:
        remasks = [remasks[i] for i in l_index]
    Generated_sequences = [Generated_sequences[i] for i in l_index]
    Proposed_sequences = [Proposed_sequences[i] for i in l_index]
    changes = [changes[i] for i in l_index]
    levenshtein = [levenshtein[i] for i in l_index]
    unmaskLogProbs = [unmaskLogProbs[i] for i in l_index]
    entropies = [entropies[i] for i in l_index]
    attention_masks = attention_masks[l_index]
    if H is not None:
        H = [H[i] for i in l_index]
    entropic_costs = [entropic_costs[i] for i in l_index]


    return logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs, 


def PlotInfo(listIndex, tokenizer, title_list, save_path=f"PipelineTest/PlotResults",  ouputpath='PipelineTest/results/outputs.pt'):
    logprobs, masks, remasks, Generated_sequences, Proposed_sequences, changes, levenshtein, unmaskLogProbs, attention_masks, H, block_size, entropies, entropic_costs = GetInfoFromIndex(listIndex, tokenizer, ouputpath)
    getTxt(Proposed_sequences, save_path, proposed_sequence=True)
    getTxt(Generated_sequences, save_path, proposed_sequence=False)
    plotLogProbs(logprobs, title_list, save_path, block_size, Generated_sequences, masks)
    plotEntropy(entropies, title_list, save_path, block_size, Generated_sequences, masks)
    plotMasks(masks, title_list, save_path, block_size, Generated_sequences)
    plotChanges(changes, title_list, save_path, block_size, Generated_sequences)
    PlotlyLogProbs(logprobs, title_list, save_path, block_size, Proposed_sequences, masks, entropies=entropies)
    PlotlyEntropy(entropies, title_list, save_path, block_size, Proposed_sequences, masks,)


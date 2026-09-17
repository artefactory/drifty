from GenerateWithMDLMSampler import CreateBaseSampleWithHistory
import dllm
from dataclasses import dataclass
from GetInfoFromBaseSamplerOutput import getEntropy, getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getUnmaskLogProbs
from PlotResults import PlotlyH, plotH, plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotNumTransferredTokens, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges , getTxt
from dllm.pipelines.dream import DreamSamplerWithCompleteHistory
from dllm.pipelines.diffusiongemma.sampler import DiffusionGemmaSamplerWithCompleteHistory, DiffusionGemmaSamplerConfig

import torch
import transformers
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion


def run_experiment_MDML(model_cfg, messages, SamplerConfig):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )

    outputs, tokenizer = CreateBaseSampleWithHistory(dllm.core.samplers.MDLMSamplerWithCompleteHistory,messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()


    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, remasking=False)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], )


    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences, masks,)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences,)



def run_experiment_MDML_remasking(model_cfg, messages, SamplerConfig):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )
    outputs, tokenizer = CreateBaseSampleWithHistory(dllm.core.samplers.MDLMSamplerRemaskingWithCompleteHistory, messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    remasks = getEachStepMask(outputs, remask=True)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    H = getH(outputs)
    num_transferred_tokens = getNumTransferTokens(outputs)

    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'] , outputs.block_size, Generated_sequences, masks, remasks)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'] , outputs.block_size, Generated_sequences, remasking=False)
    plotMasks(remasks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, remasking=True)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], )
    plotH(H, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    plotNumTransferredTokens(num_transferred_tokens, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'])

    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences, masks,remasks)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences,)
    PlotlyH(H, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)


def run_experiment_Dream(model_cfg, messages, SamplerConfig):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )

    outputs, tokenizer = CreateBaseSampleWithHistory(DreamSamplerWithCompleteHistory, messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    entropies = getEntropy(outputs)
    H = getH(outputs)

    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, remasking=False)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], )
    plotH(H, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)

    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences, masks,entropies =entropies)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences,)


def run_experiment_DiffusionGemma(model_cfg, messages, SamplerConfigClass=None):
    """Run DiffusionGemma experiment with trajectory tracking.

    Args:
        model_cfg: dict with keys 'path', 'name', 'dir', and optionally 'device_map'.
        messages: list of conversations [[{"role": "user", "content": ...}], ...]
        SamplerConfigClass: dataclass inheriting DiffusionGemmaSamplerConfig, or None for defaults.
    """
    model_path = model_cfg["path"]
    torch_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_path)

    # Build device_map: split encoder+decoder layers across 2 GPUs
    if "device_map" in model_cfg:
        device_map = model_cfg["device_map"]
    else:
        device_map = {
            "model.encoder.language_model.embed_tokens": 0,
            "model.decoder.embed_tokens": 0,
            "model.encoder.vision_tower": 0,
            "model.encoder.embed_vision": 0,
            "model.decoder.self_conditioning": 0,
            "lm_head": 0,
            "model.encoder.language_model.norm": 1,
            "model.decoder.norm": 1,
        }
        for i in range(30):
            gpu = 0 if i < 15 else 1
            device_map[f"model.encoder.language_model.layers.{i}"] = gpu
            device_map[f"model.decoder.layers.{i}"] = gpu
            
        device_map["model.encoder.language_model.norm"] = 1
        device_map["model.decoder.norm"] = 1

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device_map,
    ).eval()

    sampler = DiffusionGemmaSamplerWithCompleteHistory(
        model=model, tokenizer=processor.tokenizer,
    )

    if SamplerConfigClass is not None:
        sampler_config = SamplerConfigClass()
    else:
        sampler_config = DiffusionGemmaSamplerConfig(return_dict=True)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )

    outputs = sampler.sample(inputs, sampler_config)
    tokenizer = processor.tokenizer

    # Extract trajectories
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    entropies = getEntropy(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()


    labels = [f"{model_cfg['name']} - EX{i+1}" for i in range(len(messages))]

    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, labels, model_cfg['dir'], outputs.block_size, Generated_sequences, masks, )
    plotMasks(masks, labels, model_cfg['dir'], outputs.block_size, Generated_sequences, remasking=False)
    plotChanges(changes, labels, model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotLevenshtein(levenshtein, labels, model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, labels, model_cfg['dir'])

    PlotlyLogProbs(logprobs, labels, model_cfg['dir'], outputs.block_size, Proposed_sequences, masks, entropies=entropies)
    PlotlyChanges(changes, labels, model_cfg['dir'], outputs.block_size, Proposed_sequences)

    return outputs, tokenizer

"""
python -u examples/llada/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm
import torch


def CreateBaseSampleWithHistory(
    sampler_type,
    messages: list[list[dict[str, str]]],
    config,
    Script,
    seed_override: int | None = None,
):
    
    parser = transformers.HfArgumentParser((Script, config))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    if seed_override is not None:
        seed = int(seed_override) & ((1 << 32) - 1)
        transformers.set_seed(seed)
    elif script_args.seed is not None:
        transformers.set_seed(script_args.seed)
    else:
        seed = int(torch.seed()) & ((1 << 32) - 1)
        print('seed', seed)
        transformers.set_seed(seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = sampler_type(model=model, tokenizer=tokenizer)

    terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)
    inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
    sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    

    

    if script_args.visualize:
        terminal_visualizer.visualize(outputs.histories_x, rich=True)
    

    return outputs, tokenizer






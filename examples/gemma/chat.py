"""
Interactive chat / sampling script for DiffusionGemma.

DiffusionGemma uses its own built-in block-diffusion generation via model.generate().
It does NOT use dllm's MDLMSampler — the diffusion loop is internal to the model.

Examples
--------
python -u examples/gemma/chat.py
"""

import sys
from dataclasses import dataclass
import torch
import transformers

try:
    from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
except ImportError:
    raise ImportError(
        "DiffusionGemma requires transformers from source: "
        "pip install git+https://github.com/huggingface/transformers.git"
    )


@dataclass
class ScriptArguments:
    model_name_or_path: str = "google/diffusiongemma-26B-A4B-it"
    seed: int = 42
    max_new_tokens: int = 512

    def __post_init__(self):
        try:
            import dllm
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )
        except Exception:
            pass


def main():
    parser = transformers.HfArgumentParser((ScriptArguments,))
    (script_args,) = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    print(f"Loading model: {script_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(script_args.model_name_or_path)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        script_args.model_name_or_path,
        dtype="auto",
        device_map="auto",
    )
    print("Model loaded. Type 'quit' to exit.\n")

    conversation = []

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        input_ids = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=script_args.max_new_tokens,
            )

        # Decode only the generated part
        generated_ids = output[0][input_ids["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        print(f"Assistant: {response}\n")
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)

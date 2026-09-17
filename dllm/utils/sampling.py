import torch


def sample_trim(tokenizer, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids: Full sequence token ids from the model (prompt + generation).
        input_ids: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    # Make sure we can index these
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        prompt = list(input_ids)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)

        # Resolve the real end-of-turn token id.
        # eot_token_id is not a standard HF attribute; try to resolve it from the
        # vocabulary so that <|eot_id|> (LLaMA/LLaDA chat) takes priority over
        # <|endoftext|> which can appear mid-generation without ending the answer.
        eot_id = getattr(tokenizer, "eot_token_id", None)
        if eot_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            _candidate = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(_candidate, int) and _candidate != unk_id:
                eot_id = _candidate

        # Skip left padding only when PAD is distinct from EOS/EOT.
        # Chat templates legitimately start with EOS-like control tokens.
        if pad_id is not None and pad_id not in (eos_id, eot_id):
            while full and full[0] == pad_id:
                full.pop(0)
            while prompt and prompt[0] == pad_id:
                prompt.pop(0)
            start = len(prompt)
        else:
            # Left-padding may use eos_token_id (e.g. Dream); locate the prompt
            # subsequence in the output to find the correct generation start.
            start = len(prompt)  # fallback (works when no padding)
            if len(full) > len(prompt):
                for offset in range(len(full) - len(prompt) + 1):
                    if full[offset:offset + len(prompt)] == prompt:
                        start = offset + len(prompt)
                        break

        end = len(full)

        # For chat models, EOT is the true assistant stop token.
        # EOS can appear mid-generation (<|endoftext|>) without meaning end of answer.
        stop_ids: set[int] = set()
        if eot_id is not None:
            stop_ids.add(eot_id)
        elif eos_id is not None:
            stop_ids.add(eos_id)

        for i in range(start, len(full)):
            if full[i] in stop_ids:
                end = i
                break

        gen_ids = full[start:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        sequences.append(text)
    return sequences


def infill_trim(tokenizer, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids: Full sequence token ids from the model (prompt + generation).
        input_ids: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    # Make sure we can index these
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = torch.tensor(seq_ids)
        prompt = torch.tensor(input_ids)

        # Skip left padding tokens (necessary for dream)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eot_id = getattr(tokenizer, "eot_token_id", None)
        if pad_id is not None and pad_id not in (eos_id, eot_id):
            while full.numel() and full[0].item() == pad_id:
                full = full[1:]
            while prompt.numel() and prompt[0].item() == pad_id:
                prompt = prompt[1:]

        masked_index = prompt == tokenizer.mask_token_id
        infill = full[masked_index]

        end = len(infill)

        if eot_id is not None:
            for i in range(len(infill)):
                if infill[i] == eot_id:
                    end = i
                    break
        elif eos_id is not None:
            for i in range(len(infill)):
                if infill[i] == eos_id:
                    end = i
                    break

        gen_ids = infill[:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # in case there is no eos_id or eot_id, just strings
        eos = getattr(tokenizer, "eos_token", None)
        eot = getattr(tokenizer, "eot_token", None)
        if eos:
            text = text.split(eos)[0]
        if eot:
            text = text.split(eot)[0]
        # return text.strip()
        sequences.append(text)
    return sequences

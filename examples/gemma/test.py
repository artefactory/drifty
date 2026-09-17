from transformers import TextDiffusionStreamer
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor



streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)

message = [
    {"role": "user", "content": "Why is the sky blue?"}
]

input_ids = processor.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**input_ids, max_new_tokens=512, streamer=streamer)

text = processor.decode(output[0], skip_special_tokens=False)
print("\n-- Final Output --")
print(text[0])  
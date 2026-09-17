"""
scripts/eval_configs.py
=======================
Centralized evaluation configurations.

Pure data module (no heavy dependencies) so it can be imported anywhere
without pulling in sklearn / the whole feature stack.
"""

CONFIGS = {
    # =========================================================================
    # CONFIGURATIONS : 16 STEPS / 32 TOKENS (Existantes)
    # =========================================================================
    "dream16_32tokens_32samples_triviaqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_32samples_seed42/outputs_dream_16steps_32tokens_triviaqa_32samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_triviaqa_32samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_triviaqa_32samples_seed42/tokenizer_dream_16steps_32tokens_triviaqa_32samples_seed42.pt",
        "name": "dream_16steps_32tokens_triviaqa_32samples_evalqwen_seed42",
    },
     "dream16_32tokens_32samples_naturalquestion_evalqwen_seed42": {
            "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_32samples_seed42/outputs_dream_16steps_32tokens_naturalquestion_32samples_seed42.pt",
            "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_naturalquestion_32samples_seed42.json",
            "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_naturalquestion_32samples_seed42/tokenizer_dream_16steps_32tokens_naturalquestion_32samples_seed42.pt",
            "name": "dream_16steps_32tokens_naturalquestion_32samples_evalqwen_seed42",
    },
    "dream16_32tokens_32samples_hotpotqa_evalqwen_seed42": {
        "outputs_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_32samples_seed42/outputs_dream_16steps_32tokens_hotpotqa_32samples_seed42.pt",
        "eval_json": "PipelineTest/res/eval_V2/results_dream_16steps_32tokens_hotpotqa_32samples_seed42.json",
        "tokenizer_path": "PipelineTest/res/results_dream_16steps_32tokens_hotpotqa_32samples_seed42/tokenizer_dream_16steps_32tokens_hotpotqa_32samples_seed42.pt",
        "name": "dream_16steps_32tokens_hotpotqa_32samples_evalqwen_seed42",
    },

}


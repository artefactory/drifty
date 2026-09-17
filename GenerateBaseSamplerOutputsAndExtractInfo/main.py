from GenerateWithMDLMSampler import CreateBaseSampleWithHistory
import dllm
from dataclasses import dataclass
from GetInfoFromBaseSamplerOutput import getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getUnmaskLogProbs
from PlotResults import PlotlyH, plotH, plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotNumTransferredTokens, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges , getTxt
from run_experiments import run_experiment_MDML, run_experiment_MDML_remasking, run_experiment_Dream, run_experiment_DiffusionGemma
from dllm.pipelines.diffusiongemma.sampler import DiffusionGemmaSamplerConfig

import sys
sys.argv = [sys.argv[0]] # Cette ligne "vide" les arguments du terminal



@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"


@dataclass
class DreamSamplerConfig(dllm.pipelines.dream.DreamSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0
    top_p: float = 1.0


@dataclass
class GemmaSamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 32
    steps: int = 16
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    max_temperature: float = None
    min_temperature: float = None
    return_dict: bool = True
    canvas_length: int = 32




messages = [
    # # 1. CODE : Test de la logique et de la syntaxe
    # [{"role": "user", "content": "Write a Python script for Fibonacci sequence."}],
    
    # # 2. CULTURE G : Test de précision factuelle (Noms, dates, lieux)
    # [{"role": "user", "content": "Which city is the capital of Australia?"}],
    
    # 3. HISTOIRE : Test de créativité et de cohérence narrative (longue traîne)
    [{"role": "user", "content": "Which theory states that 'people tend to rise to their own level of incompetence'? Please answer briefly and concisely."}],
]


MODELS_TO_TEST = [
    # {
    #     "name": "ModernBERT",
    #     "path": "dllm-hub/ModernBERT-large-chat-v0.1",
    #     "dir": "ModernBERT"
    # },
    # {
    #     "name": "LLaDa",
    #     "path": "GSAI-ML/LLaDA-8B-Instruct", 
    #     "dir": "StudyAutoRegressionBehavior/Results/LLaDa_8B_remasking_1Block",
    #     "remasking": True
    # },
    # {
    #     "name": "LLaDa",
    #     "path": "GSAI-ML/LLaDA-8B-Instruct", 
    #     "dir": "StudyAutoRegressionBehavior/Results/LLaDa_8B_no_remasking_1Block",
    #     "remasking": False
    # },
    # {
    #     "name": "Qwen-MDLM",
    #     "path": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
    #     "dir": "Qwen_MDLM"
    # },
    # {
    #     "name": "a2d",
    #     "path":  "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    #     "dir": "a2d"
        
    # },

    # {
    #     "name": "Dream",
    #     "path": "Dream-org/Dream-v0-Instruct-7B",
    #     "dir": "GenerateBaseSamplerOutputsAndExtractInfo/Results/Dream",
    #     "type": "dream"
    # },

    # {
    #     "name": "editflow",
    #     "path": "./.models/editflow/ModernBERT-large/alpaca/checkpoint-final",
    #     "dir": "editflow"
        
    # },
    # {
    #     "name": "LargeBERT",
    #     "path": "dllm-hub/ModernBERT-large-chat-v0.1",
    #     "dir": "LargeBERT"
        
    # },

    # {
    #     "name": "qwen3MDML",
    #     "path": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    #     "dir": "qwen3MDML"
        
    # },

   
    {
        "name": "DiffusionGemma",
        "path": "google/diffusiongemma-26B-A4B-it",
        "dir": "GenerateBaseSamplerOutputsAndExtractInfo/Results/DiffusionGemma",
        "type": "diffusiongemma"
    },

]

def main():
    for model_cfg in MODELS_TO_TEST:
        # try:
            model_type = model_cfg.get('type', 'mdlm',)
            if model_type == 'dream':
                run_experiment_Dream(model_cfg, messages, DreamSamplerConfig)
            elif model_type == 'diffusiongemma':
                run_experiment_DiffusionGemma(model_cfg, messages, GemmaSamplerConfig)
            elif model_cfg.get('remasking', False):  
                run_experiment_MDML_remasking(model_cfg, messages, SamplerConfig) 
            else:
                run_experiment_MDML(model_cfg, messages, SamplerConfig)
        # except Exception as e:
        #     print(f"❌ Error with model {model_cfg['name']}: {e}")
        #     continue



if __name__ == "__main__":
    main()

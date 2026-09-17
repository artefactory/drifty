"""
PipelineTest.features
=====================
Feature extraction modules for hallucination detection in discrete diffusion LMs.

Modules:
- utils: shared utilities (merge outputs, padding detection, labels)
- baseline: simple mean/variance features
- markovian: dynamic features motivated by the Markovian model (alpha, beta, C, tau, m)
- ar1: AR(1) model fitting and derived features
- semantic: semantic token clustering and semantic entropy

Usage:
    from PipelineTest.features.utils import load_outputs, get_labels
    from PipelineTest.features.baseline import get_baseline_features
    from PipelineTest.features.markovian import get_markovian_features
    from PipelineTest.features.ar1 import fit_ar1_models, get_ar1_features
    from PipelineTest.features.semantic import get_semantic_features
"""

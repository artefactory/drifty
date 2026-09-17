# PipelineTest/vllm_utils

This folder launches the **vLLM** server of the LLM judge used by [`sampling/eval.py`](../sampling/README.md#step-2--evaluation-evalpy) (step 2).

vLLM pins its own `torch` and `transformers` versions, which are incompatible with the `lladadream` and `diffugemma` environments. It therefore lives in a **separate uv project** (by default `../eval_server`, next to the repository). The pipeline only talks to it over HTTP, through its OpenAI-compatible API.

## `VLLMServerManager`

```python
from PipelineTest.vllm_utils.vllm_server_manager import VLLMServerManager

with VLLMServerManager(
    model_name_or_path="Qwen/Qwen3.5-9B",
    eval_server_dir="/path/to/eval_server",
    port=8000,
    extra_args=["--trust-remote-code"],
    log_path="vllm_server.log",
) as server:
    client = OpenAI(base_url=server.base_url, api_key="not-needed")
    ...
```

- **`start()`:** runs `uv run --project <eval_server_dir> vllm serve <model> --port … --max-model-len … --gpu-memory-utilization … --dtype …` in a new process group. The environment is cleaned: only `HOME`, `USER`, `SHELL`, `TERM` and `LANG` are kept, and `dllm` entries are removed from `PATH`, so the project venv does not leak in. `start()` then polls `/health` until the server responds.
- **`stop()`:** sends `SIGTERM` to the group, then `SIGKILL` after 30 s.
- **`base_url`:** `http://localhost:<port>/v1`.

| Parameter | Default |
|-----------|---------|
| `port` | `8123` |
| `max_model_len` | `8192` |
| `gpu_memory_utilization` | `0.85` |
| `dtype` | `bfloat16` |
| `cuda_visible_devices` | inherited (unchanged) |
| `startup_timeout` | `600` s |
| `log_path` | `None` (output discarded) |

## Setting up `eval_server`

`eval_server` is a minimal uv project (`requires-python >= 3.11`) whose only dependency is `vllm==0.19.1`:

```bash
cd ../eval_server
uv sync
uv run vllm serve Qwen/Qwen3.5-9B --port 8000   # manual test
```

import os
import signal
import subprocess
import time
import requests


class VLLMServerManager:
    def __init__(
        self,
        model_name_or_path: str,
        eval_server_dir: str,
        port: int = 8123,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        cuda_visible_devices: str | None = None,
        startup_timeout: int = 600,
        extra_args: list[str] | None = None,
        log_path: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.eval_server_dir = eval_server_dir
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.cuda_visible_devices = cuda_visible_devices
        self.startup_timeout = startup_timeout
        self.extra_args = extra_args or []
        self.log_path = log_path
        self.process = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self):
        cmd = [
            "uv", "run", "--project", self.eval_server_dir,
            "vllm", "serve", self.model_name_or_path,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
        ] + self.extra_args

        keep_vars = ["HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL"]
        env = {k: os.environ[k] for k in keep_vars if k in os.environ}

        # Nettoie le PATH : retire tout ce qui vient d'un venv "dllm"
        clean_path_entries = [
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "dllm" not in p
        ]
        env["PATH"] = os.pathsep.join(clean_path_entries)

        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        import shutil
        resolved_uv = shutil.which("uv", path=env["PATH"])
        print("[vLLM DEBUG] uv résolu vers:", resolved_uv, flush=True)
        print("[vLLM DEBUG] PATH nettoyé:", env["PATH"], flush=True)

        log_file = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self.process = subprocess.Popen(
            cmd, env=env, cwd=self.eval_server_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._wait_until_ready()

    def _wait_until_ready(self):
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (code={self.process.returncode}). "
                    f"Check log at {self.log_path}"
                )
            try:
                r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    print(f"[vLLM] Server ready on port {self.port} after {time.time()-t0:.1f}s", flush=True)
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM server did not become ready within {self.startup_timeout}s")

    def stop(self):
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
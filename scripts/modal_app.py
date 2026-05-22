"""Modal app: persistent T4 container that loads a 7-8B LLM in 4-bit and runs batched inference.

Design notes:
- One Modal class. Container stays warm across calls; `_loaded_id` tracks which model is in VRAM.
- Switching models inside one container is cheaper than spinning up new containers (saves ~30s of cold-start each).
- HF weights are cached in a Modal Volume so the multi-GB download is paid only once across the whole project.
- HF token (if present) is read from a Modal Secret named "huggingface" with key HF_TOKEN.

Usage from a driver script (see `scripts/run_llm_matrix.py`):
    import modal
    app = modal.App.lookup("finllama-sentiment")
    runner = modal.Cls.lookup("finllama-sentiment", "LLMRunner")()
    raw_outputs = runner.generate.remote(hf_id="...", prompts=[...], max_new_tokens=20, batch_size=8)
"""

from __future__ import annotations

import modal

app = modal.App("finllama-sentiment")

# Persistent volume for HF weights — pay download cost only once.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Container image: torch + transformers + bitsandbytes for 4-bit loading.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        # 4.55 adds Qwen3 + Gemma-2 support. Pinning only transformers and
        # letting pip resolve compatible tokenizers/accelerate/bitsandbytes
        # avoids version-pin churn each time we bump models.
        "transformers==4.55.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.3",
        "huggingface_hub>=0.26.0",
        "sentencepiece==0.2.0",       # required by Mistral / some tokenizers
        "protobuf==4.25.4",
    )
    .env({"HF_HOME": "/cache/hf", "TRANSFORMERS_CACHE": "/cache/hf"})
)

# Optional secret. If not configured, container runs without HF_TOKEN
# (fine for ungated models like Qwen, Mistral, FinLLaMA).
try:
    hf_secret = modal.Secret.from_name("huggingface")
    SECRETS = [hf_secret]
except modal.exception.NotFoundError:
    SECRETS = []


@app.cls(
    gpu="T4",
    image=image,
    volumes={"/cache": hf_cache},
    secrets=SECRETS,
    timeout=60 * 30,           # 30 min per call ceiling
    scaledown_window=120,      # idle 2 min then shut down
    max_containers=3,          # allow parallel runs across 3 containers
)
class LLMRunner:
    """Persistent T4 container that loads/swaps 7-8B LLMs in 4-bit and runs inference."""

    @modal.enter()
    def setup(self):
        # State carried across .generate calls in the same container.
        self._loaded_id: str | None = None
        self._tokenizer = None
        self._model = None

    def _load(self, hf_id: str):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # Free previous model if any.
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            import gc; gc.collect()
            torch.cuda.empty_cache()

        # Explicit token pass (belt and suspenders — also picked up from HF_HOME env).
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("[modal] HF_TOKEN present (length=%d)" % len(hf_token))
        else:
            print("[modal] HF_TOKEN not set — only public models will load")

        print(f"[modal] loading tokenizer: {hf_id}")
        tok = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        print(f"[modal] loading model in 4-bit: {hf_id}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map="auto", token=hf_token,
        )
        model.eval()

        self._tokenizer = tok
        self._model = model
        self._loaded_id = hf_id
        print(f"[modal] ready: {hf_id}")

    @modal.method()
    def generate(
        self,
        hf_id: str,
        prompts: list[str],
        max_new_tokens: int = 20,
        batch_size: int = 8,
        use_chat_template: bool = False,
    ) -> list[str]:
        """Generate one continuation per prompt. Returns just the new tokens decoded.

        Greedy (do_sample=False, deterministic). Latency is reported by caller from wall time.

        If `use_chat_template=True` and the tokenizer ships a chat template (Llama-3 / Qwen / Mistral
        instruct flavours), the prompt is wrapped as a single user-turn before tokenization. This is
        required for some instruction-tuned models (e.g. plutus-8B-instruct) that otherwise emit EOS
        immediately when handed a raw classification prompt.
        """
        import time
        import torch

        if self._loaded_id != hf_id:
            self._load(hf_id)

        # Optional chat-template wrap.
        chat_tpl = getattr(self._tokenizer, "chat_template", None)
        if use_chat_template and chat_tpl:
            wrapped = []
            # Qwen3's chat template defaults to enable_thinking=True, which
            # emits a <think>...</think> block before the answer and chews
            # through max_new_tokens. We disable it for classification/NER
            # where we want a single short JSON answer. apply_chat_template
            # silently ignores unknown kwargs on models that don't support it.
            extra_kwargs = {}
            try:
                # Best-effort feature-detect: peek at template source for "enable_thinking".
                if "enable_thinking" in (chat_tpl or ""):
                    extra_kwargs["enable_thinking"] = False
            except Exception:
                pass
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                try:
                    wrapped.append(
                        self._tokenizer.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True,
                            **extra_kwargs,
                        )
                    )
                except TypeError:
                    # Template didn't accept the kwarg — fall back to plain call.
                    wrapped.append(
                        self._tokenizer.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                    )
            prompts = wrapped
            print(f"[modal] applied chat template to {len(prompts)} prompts "
                  f"(extra={extra_kwargs})")

        out: list[str] = []
        n = len(prompts)
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            batch = prompts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self._model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            for row in ids:
                new = row[input_len:]
                out.append(self._tokenizer.decode(new, skip_special_tokens=True).strip())

        wall = time.perf_counter() - t0
        print(f"[modal] {hf_id}: generated {n} in {wall:.1f}s ({wall/max(n,1)*1000:.0f}ms/sample)")
        return out


@app.local_entrypoint()
def smoke(hf_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Tiny end-to-end check: load a model, run 5 prompts, print outputs."""
    runner = LLMRunner()
    prompts = [
        "Classify the sentiment of: Apple beat earnings expectations and raised guidance.\nSentiment:",
        "Classify the sentiment of: The company filed for bankruptcy.\nSentiment:",
        "Classify the sentiment of: Quarterly revenue was in line with analyst estimates.\nSentiment:",
        "Classify the sentiment of: The CEO resigned amid an SEC investigation.\nSentiment:",
        "Classify the sentiment of: Shares ticked up 0.2% in afternoon trade.\nSentiment:",
    ]
    out = runner.generate.remote(
        hf_id=hf_id,
        prompts=prompts,
        max_new_tokens=15,
        batch_size=5,
    )
    for p, o in zip(prompts, out):
        print("PROMPT:", p[:80].replace("\n", " "))
        print("OUTPUT:", o)
        print()

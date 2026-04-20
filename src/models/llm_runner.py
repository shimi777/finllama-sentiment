"""Inference wrapper for FinLLaMA-Instruct and LLaMA-3.1-8B-Instruct (4-bit)."""

from __future__ import annotations

import time
from tqdm import tqdm

from src.utils import get_logger, set_seed

logger = get_logger(__name__)


class LLMRunner:
    """Load an 8B-class LLM in 4-bit and run batched text generation."""

    def __init__(self, hf_id: str, load_in_4bit: bool = True, seed: int = 42):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.hf_id = hf_id
        set_seed(seed)

        logger.info("Loading tokenizer: %s", hf_id)
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # required for batched generation

        logger.info("Loading model (4-bit=%s): %s", load_in_4bit, hf_id)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                hf_id, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.float16, device_map="auto"
            )
        self._model.eval()
        logger.info("Model loaded.")

    def generate(
        self,
        prompts: list[str],
        batch_size: int = 8,
        max_new_tokens: int = 20,
    ) -> list[tuple[str, float]]:
        """Generate a response for each prompt.

        Returns:
            List of (raw_output, latency_ms) tuples in input order.
            raw_output contains only the newly generated tokens (not the prompt).
        """
        import torch

        results: list[tuple[str, float]] = []
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        for batch in tqdm(batches, desc=f"Generating ({self.hf_id.split('/')[-1]})"):
            t0 = time.perf_counter()
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self._model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,    # ignored when do_sample=False
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            latency_ms = (time.perf_counter() - t0) * 1000 / len(batch)

            for ids in output_ids:
                new_tokens = ids[input_len:]
                text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                results.append((text, latency_ms))

        return results

    def unload(self) -> None:
        """Free GPU memory before loading the next model."""
        import gc
        import torch

        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LLMRunner unloaded: %s", self.hf_id)

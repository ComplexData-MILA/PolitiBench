"""Lazy local model loading shared by feasibility and Qwen verdict evaluation."""

import os

from evaluation_utils import load_environment, repo_path


def resolve_device(requested):
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


class LocalTextModel:
    def __init__(
        self,
        model_name,
        device="auto",
        thinking=False,
        max_new_tokens=None,
        seed=42,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.device = None
        self.model = None
        self.tokenizer = None
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens or (1024 if thinking else 16)
        self.seed = seed

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging

        load_environment()
        hf_logging.set_verbosity_error()
        self.device = resolve_device(self.requested_device)
        model_name = self.model_name
        if repo_path(model_name).is_dir():
            model_name = str(repo_path(model_name))
        hf_token = os.getenv("HF_TOKEN")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=dtype_for_device(self.device),
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print(
            f"Loaded {model_name}; thinking={self.thinking}, "
            f"max_new_tokens={self.max_new_tokens}"
        )

    def generate(self, messages):
        import torch

        self._load()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.thinking,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])

        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and input_tokens + self.max_new_tokens > max_positions:
            raise ValueError(
                f"Prompt has {input_tokens} tokens and generation allows "
                f"{self.max_new_tokens}, exceeding the model context of "
                f"{max_positions} tokens."
            )

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.thinking:
            torch.manual_seed(self.seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            generation_kwargs.update(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = outputs[0, input_tokens:]
        output_tokens = int(generated_ids.numel())
        output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        usage = {
            "resolved_model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "attempts": 1,
        }
        return output, usage


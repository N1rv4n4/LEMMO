from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compat import configure_runtime
from .io import (
    EMLM_DESCRIPTION_MAXIMUM_LENGTH,
    EMLM_DESCRIPTION_MINIMUM_LENGTH,
    EMLM_RECOGNITION_LENGTHS,
    load_iq,
    resolve_path,
)
from .lora import inject_qwen_lora
from .models import EMLMRecognitionModel, EMLMDescriptionModel, load_emlm_recognition_weights, load_emlm_description_weights


EMLM_RECOGNITION_SYSTEM_PROMPT = (
    "你是电磁信号分析助手。请仅依据提供的电磁信号特征简洁回答问题；"
    "如果证据不足，应明确说明无法判断。"
)
EMLM_DESCRIPTION_SYSTEM_PROMPT = (
    "You are an electromagnetic signal analysis assistant. Analyze the provided "
    "IQ signal and answer the user's question accurately and concisely."
)


def _model_reference(config_path: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    local = candidate if candidate.is_absolute() else config_path.parent.parent / candidate
    return str(local.resolve()) if local.exists() else value


def _prompt(tokenizer, system_prompt: str, question: str, device: torch.device) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endings = torch.nonzero(input_ids[0].eq(im_end), as_tuple=False).flatten()
    if endings.numel() < 2:
        raise RuntimeError("Could not locate the user message boundary")
    question_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if not question_ids.numel():
        raise ValueError("Question must not be empty")
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "question_ids": question_ids,
        "question_mask": torch.ones_like(question_ids, dtype=torch.bool),
        "insert_index": int(endings[-1]),
    }


def _insert_signal(embedding_layer, prompt: dict, signal_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if signal_tokens.ndim != 3 or signal_tokens.shape[0] != 1:
        raise ValueError("Public CLI currently expects one signal per request")
    text = embedding_layer(prompt["input_ids"])
    index = prompt["insert_index"]
    combined = torch.cat((text[:, :index], signal_tokens, text[:, index:]), dim=1)
    attention = torch.ones(combined.shape[:2], dtype=torch.bool, device=combined.device)
    return combined, attention


def _greedy_generate(
    qwen,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    maximum_new_tokens: int,
) -> str:
    generated: list[int] = []
    eos = int(tokenizer.eos_token_id)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].float().argmax(dim=-1)
        for _ in range(int(maximum_new_tokens)):
            token = int(next_token.item())
            if token == eos:
                break
            generated.append(token)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)),
                dim=1,
            )
            output = qwen(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            next_token = output.logits[:, -1].float().argmax(dim=-1)
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


class _BaseRuntime:
    def __init__(self, config_path: str | Path, device: str = "cuda:0") -> None:
        configure_runtime()
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("EMLM inference requires a CUDA GPU")
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        reference = _model_reference(self.config_path, self.config["qwen_model"])
        local_only = Path(reference).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=local_only)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            reference,
            local_files_only=local_only,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)

    def _load_lora(self, saved: dict) -> None:
        setting = self.config["lora"]
        inject_qwen_lora(
            self.qwen,
            rank=int(setting["rank"]),
            alpha=float(setting["alpha"]),
            dropout=float(setting.get("dropout", 0.0)),
        )
        incompatible = self.qwen.load_state_dict(saved["lora_state"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected Qwen LoRA keys: {incompatible.unexpected_keys}")
        self.qwen = self.qwen.to(self.device).eval()


class EMLMRecognitionRuntime(_BaseRuntime):
    tasks = EMLMRecognitionModel.TASKS

    def __init__(self, config_path: str | Path, device: str = "cuda:0") -> None:
        super().__init__(config_path, device)
        self.model = EMLMRecognitionModel(self.config)
        saved = load_emlm_recognition_weights(
            self.model, resolve_path(self.config_path, self.config["inference_checkpoint"])
        )
        self._load_lora(saved)
        self.model = self.model.to(self.device).eval()

    def answer(
        self,
        signal_path: str | Path,
        sample_rate_hz: float,
        task: str,
        question: str,
        maximum_new_tokens: int = 64,
        system_prompt: str = EMLM_RECOGNITION_SYSTEM_PROMPT,
    ) -> str:
        if task not in self.tasks:
            raise ValueError(f"task must be one of {self.tasks}")
        array = load_iq(signal_path, EMLM_RECOGNITION_LENGTHS)
        signal = torch.from_numpy(array)[None].to(self.device, dtype=torch.bfloat16)
        rates = torch.tensor([sample_rate_hz], dtype=torch.float32, device=self.device)
        prompt = _prompt(self.tokenizer, system_prompt, question, self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            question_embeddings = self.qwen.get_input_embeddings()(prompt["question_ids"])
            projected = self.model(
                signal, rates, question_embeddings, prompt["question_mask"], task
            )
            embeddings, attention = _insert_signal(self.qwen.get_input_embeddings(), prompt, projected)
        return _greedy_generate(
            self.qwen, self.tokenizer, embeddings, attention, maximum_new_tokens
        )


class EMLMDescriptionRuntime(_BaseRuntime):
    def __init__(self, config_path: str | Path, device: str = "cuda:0") -> None:
        super().__init__(config_path, device)
        self.model = EMLMDescriptionModel(int(self.config.get("query_count", 64)))
        self.system_prompt = self._description_system_prompt()
        saved = load_emlm_description_weights(
            self.model, resolve_path(self.config_path, self.config["inference_checkpoint"])
        )
        self._load_lora(saved)
        self.model = self.model.to(self.device).eval()

    def _description_system_prompt(self) -> str:
        value = self.config.get("system_prompt_file")
        if value is None:
            return EMLM_DESCRIPTION_SYSTEM_PROMPT
        prompt_path = resolve_path(self.config_path, str(value))
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"System prompt is empty: {prompt_path}")
        return prompt


    def answer(
        self,
        signal_path: str | Path,
        sample_rate_hz: float,
        input_setting: str,
        question: str,
        maximum_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be a finite positive value")
        if not str(input_setting).strip():
            raise ValueError("input_setting must not be empty")
        if not str(question).strip():
            raise ValueError("question must not be empty")
        user_prompt = f"{str(input_setting).strip()}\n\n用户问题：{str(question).strip()}"
        array = load_iq(
            signal_path,
            minimum_length=EMLM_DESCRIPTION_MINIMUM_LENGTH,
            maximum_length=EMLM_DESCRIPTION_MAXIMUM_LENGTH,
        )
        signal = torch.from_numpy(array)[None].to(self.device, dtype=torch.bfloat16)
        rates = torch.tensor([sample_rate_hz], dtype=torch.float32, device=self.device)
        prompt = _prompt(
            self.tokenizer, system_prompt or self.system_prompt, user_prompt, self.device
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            question_embeddings = self.qwen.get_input_embeddings()(prompt["question_ids"])
            projected = self.model(
                signal, rates, question_embeddings, prompt["question_mask"]
            )
            embeddings, attention = _insert_signal(self.qwen.get_input_embeddings(), prompt, projected)
        return _greedy_generate(
            self.qwen, self.tokenizer, embeddings, attention, maximum_new_tokens
        )


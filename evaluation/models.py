from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class CloseSourceModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        # openai client initialization
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = dict(extra or {})
        self.extra.update(kwargs)  # allow passing extra sdk params

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        params = dict(self.extra)
        params.update(kwargs)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **params,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[CloseSourceModel] {e}")
            return ""


class OpenSourceModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_length: Optional[int] = 4096,
        trust_remote_code: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.extra = dict(extra or {})
        self.extra.update(kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        max_length = kwargs.pop("max_length", self.max_length)

        params = dict(self.extra)
        params.update(kwargs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        do_sample = temperature > 0  # sample only if needed

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **params,
                )
                
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"[OpenSourceModel] {e}")
            return ""


class ModelFactory:
    @staticmethod
    def create_model(model_type: Literal["close_source", "open_source"], **kwargs) -> BaseModel:
        if model_type == "close_source":
            return CloseSourceModel(**kwargs)
        if model_type == "open_source":
            return OpenSourceModel(**kwargs)
        raise ValueError(f"Unknown model_type: {model_type}")
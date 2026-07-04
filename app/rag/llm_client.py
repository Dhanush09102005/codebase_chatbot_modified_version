"""
LLM client interfaces and implementations.

Provides a common interface for different LLM providers while
maintaining clean separation of responsibilities.

The LLM client ONLY communicates with language models.
It does NOT handle retrieval, prompt building, or orchestration.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass

from app.core import logger, settings

@dataclass
class LLMResponse:
    """
    Response from an LLM.
    
    Standardized response format across all LLM providers.
    """
    text: str  # Generated response text
    model: str  # Model name used
    tokens_used: Optional[int] = None  # Total tokens (prompt + completion)
    prompt_tokens: Optional[int] = None  # Tokens in prompt
    completion_tokens: Optional[int] = None  # Tokens in completion
    time_seconds: Optional[float] = None  # Time taken for generation
    finish_reason: Optional[str] = None  # Why generation stopped
    metadata: Optional[Dict[str, Any]] = None  # Provider-specific metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "time_seconds": self.time_seconds,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata or {},
        }


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM clients must implement generate() to produce responses.
    This enables easy swapping between providers.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse: Standardized response
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass


class HuggingFaceLLMClient(LLMClient):
    """
    LLM client using HuggingFace Transformers.
    
    Supports local and remote HuggingFace models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        token: Optional[str] = None,
        max_length: int = 4096,
    ):
        """
        Initialize HuggingFace LLM client.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu" or "cuda")
            token: HuggingFace API token (for gated models)
            max_length: Maximum sequence length
        """
        model_name = model_name or settings.llm_model
        token = token or settings.hf_token
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self._model_name = model_name
            self._device = device
            self._max_length = max_length
            
            logger.info(f"Loading LLM model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=token,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            # print("=" * 60)
            # print("model_name =", model_name)
            # print("settings.hf_model_id =", settings.hf_model_id)
            # print("=" * 60)
            logger.info(f"Model loaded: {model_name}, device={device}")
            
        except ImportError:
            raise ImportError(
                "transformers or torch not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using HuggingFace model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.1)
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse: Generated response
        """
        import torch
        import time
        
        max_tokens = max_tokens or 512
        temperature = temperature if temperature is not None else 0.1
        
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length
            ).to(self._device)
            
            prompt_tokens = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            response_text = full_output[len(prompt):].strip()
            
            completion_tokens = outputs.shape[1] - prompt_tokens
            total_tokens = outputs.shape[1]
            
            elapsed = time.time() - start_time
            
            logger.info(
                f"Generated response: "
                f"prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, "
                f"time={elapsed:.2f}s"
            )
            
            return LLMResponse(
                text=response_text,
                model=self._model_name,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                time_seconds=elapsed,
                finish_reason="length" if completion_tokens >= max_tokens else "stop"
            )
        
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class GroqLLMClient(LLMClient):
    """
    LLM client using Groq API.
    
    Fast inference for Llama, Mixtral, and other models.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-8b-instant",
    ):
        """
        Initialize Groq LLM client.
        
        Args:
            api_key: Groq API key
            model_name: Model identifier (e.g., llama-3.1-8b-instant)
        """
        try:
            from groq import Groq
            
            self._model_name = model_name
            self.client = Groq(api_key=api_key)
            
            logger.info(f"Groq LLM client initialized: {model_name}")
            
        except ImportError:
            raise ImportError(
                "groq not installed. Install with: pip install groq"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Groq API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.1)
            **kwargs: Additional Groq parameters
            
        Returns:
            LLMResponse: Generated response
        """
        import time
        
        max_tokens = max_tokens or 512
        temperature = temperature if temperature is not None else 0.1
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            
            text = response.choices[0].message.content
            usage = response.usage
            
            logger.info(
                f"Groq response: "
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}, "
                f"time={elapsed:.2f}s"
            )
            
            return LLMResponse(
                text=text,
                model=self._model_name,
                tokens_used=usage.total_tokens,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                time_seconds=elapsed,
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class OpenAILLMClient(LLMClient):
    """
    LLM client using OpenAI API.
    
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4-turbo-preview",
    ):
        """
        Initialize OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Model identifier (e.g., gpt-4-turbo-preview)
        """
        try:
            from openai import OpenAI
            
            self._model_name = model_name
            self.client = OpenAI(api_key=api_key)
            
            logger.info(f"OpenAI LLM client initialized: {model_name}")
            
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.1)
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse: Generated response
        """
        import time
        
        max_tokens = max_tokens or 512
        temperature = temperature if temperature is not None else 0.1
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            elapsed = time.time() - start_time
            
            text = response.choices[0].message.content
            usage = response.usage
            
            logger.info(
                f"OpenAI response: "
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}, "
                f"time={elapsed:.2f}s"
            )
            
            return LLMResponse(
                text=text,
                model=self._model_name,
                tokens_used=usage.total_tokens,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                time_seconds=elapsed,
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


class OllamaLLMClient(LLMClient):
    """
    LLM client using Ollama.
    
    For local LLM deployment with Ollama.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model_name: Ollama model name (e.g., llama3.1)
            base_url: Ollama server URL
        """
        self._model_name = model_name
        self._base_url = base_url
        
        logger.info(f"Ollama LLM client initialized: {model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.1)
            **kwargs: Additional Ollama parameters
            
        Returns:
            LLMResponse: Generated response
        """
        import requests
        import time
        
        temperature = temperature if temperature is not None else 0.1
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                    **kwargs
                }
            )
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            
            data = response.json()
            text = data.get("response", "")
            
            logger.info(f"Ollama response: time={elapsed:.2f}s")
            
            return LLMResponse(
                text=text,
                model=self._model_name,
                time_seconds=elapsed,
                finish_reason=data.get("done_reason", "stop")
            )
        
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


def create_llm_client(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: Provider name ("huggingface", "groq", "openai", "ollama")
        model_name: Model identifier (provider-specific)
        api_key: API key (for Groq, OpenAI)
        **kwargs: Provider-specific parameters
        
    Returns:
        LLMClient: Initialized LLM client
    """
    from app.core import settings
    
    provider = provider.lower()
    
    if provider == "huggingface":
        model_name = model_name or settings.hf_model_id
        token = api_key or settings.hf_token
        device = kwargs.get("device", "cpu")
        
        return HuggingFaceLLMClient(
            model_name=model_name,
            device=device,
            token=token
        )
    
    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq requires api_key")
        model_name = model_name or "llama-3.1-8b-instant"
        return GroqLLMClient(api_key=api_key, model_name=model_name)
    
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI requires api_key")
        model_name = model_name or "gpt-4-turbo-preview"
        return OpenAILLMClient(api_key=api_key, model_name=model_name)
    
    elif provider == "ollama":
        model_name = model_name or "llama3.1"
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaLLMClient(model_name=model_name, base_url=base_url)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: huggingface, groq, openai, ollama"
        )

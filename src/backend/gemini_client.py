"""
Gemini API Client Module

This module wraps all Gemini API calls with:
- Authentication (API key from environment)
- Retry logic with exponential backoff
- Rate limiting
- Model selection (fast vs deep)
- Logging and analytics
"""

import os
import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers for different use cases."""
    FAST = "fast"      # Quick hints, short responses
    BALANCED = "balanced"  # Standard explanations
    DEEP = "deep"      # Complex lesson generation


@dataclass
class GeminiConfig:
    """Configuration for Gemini API client."""
    api_key: str
    fast_model: str = "gemini-1.5-flash"
    balanced_model: str = "gemini-1.5-flash"
    deep_model: str = "gemini-1.5-pro"
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Load config from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. API calls will fail.")
        return cls(
            api_key=api_key,
            fast_model=os.getenv("GEMINI_FAST_MODEL", "gemini-1.5-flash"),
            balanced_model=os.getenv("GEMINI_BALANCED_MODEL", "gemini-1.5-flash"),
            deep_model=os.getenv("GEMINI_DEEP_MODEL", "gemini-1.5-pro"),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "30"))
        )


@dataclass
class GeminiResponse:
    """Standardized response from Gemini API."""
    success: bool
    content: str
    model: str
    prompt_hash: str
    tokens_used: int
    latency_ms: float
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class CallMetadata:
    """Metadata logged for each Gemini call."""
    timestamp: str
    prompt_hash: str
    model: str
    tokens_used: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class CircuitBreaker:
    """
    Circuit breaker to stop calls when Gemini has repeated failures.
    Think of it like a safety switch that turns off when there are too many problems.
    """
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.is_open = False
    
    def record_success(self):
        """Reset failures on success."""
        self.failures = 0
        self.is_open = False
    
    def record_failure(self):
        """Track failure and possibly open circuit."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.warning("Circuit breaker opened due to repeated failures.")
    
    def can_proceed(self) -> bool:
        """Check if we should try a call."""
        if not self.is_open:
            return True
        # Check if reset timeout passed
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failures = 0
            logger.info("Circuit breaker reset after timeout.")
            return True
        return False


class GeminiClient:
    """
    Main client for calling Gemini API.
    
    This handles:
    - Picking the right model for the job
    - Retrying if something goes wrong
    - Logging for analytics
    - Rate limiting to avoid overuse
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        """Initialize the client with config."""
        self.config = config or GeminiConfig.from_env()
        self.circuit_breaker = CircuitBreaker()
        self.call_history: List[CallMetadata] = []
        self._genai = None
        self._models: Dict[str, Any] = {}
        
    def _init_genai(self):
        """Initialize Google Generative AI library lazily."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
    
    def _get_model(self, tier: ModelTier):
        """Get the model instance for a tier."""
        self._init_genai()
        
        model_name = {
            ModelTier.FAST: self.config.fast_model,
            ModelTier.BALANCED: self.config.balanced_model,
            ModelTier.DEEP: self.config.deep_model
        }[tier]
        
        if model_name not in self._models:
            self._models[model_name] = self._genai.GenerativeModel(model_name)
        
        return self._models[model_name], model_name
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a short hash of the prompt for logging."""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.
        Like waiting longer each time you knock on a door that doesn't open.
        """
        delay = min(
            self.config.base_delay * (2 ** attempt),
            self.config.max_delay
        )
        # Add jitter (random variation) to prevent thundering herd
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
    
    def _log_call(self, metadata: CallMetadata):
        """Log call metadata for analytics."""
        self.call_history.append(metadata)
        logger.info(
            f"Gemini call: model={metadata.model}, "
            f"tokens={metadata.tokens_used}, "
            f"latency={metadata.latency_ms:.0f}ms, "
            f"success={metadata.success}"
        )
    
    def call(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.BALANCED,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> GeminiResponse:
        """
        Call Gemini API with retry logic.
        
        Args:
            prompt: The prompt to send to Gemini
            tier: Which model tier to use (FAST, BALANCED, DEEP)
            system_instruction: Optional system prompt
            temperature: Creativity level (0=focused, 1=creative)
            max_tokens: Maximum response length
            
        Returns:
            GeminiResponse with the result or error
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            return GeminiResponse(
                success=False,
                content="",
                model="",
                prompt_hash=self._hash_prompt(prompt),
                tokens_used=0,
                latency_ms=0,
                error="Circuit breaker is open. Too many recent failures."
            )
        
        prompt_hash = self._hash_prompt(prompt)
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                model, model_name = self._get_model(tier)
                
                # Build generation config
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                
                # Make the API call
                if system_instruction:
                    # Create model with system instruction
                    model = self._genai.GenerativeModel(
                        model_name,
                        system_instruction=system_instruction
                    )
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract response text
                content = response.text if response.text else ""
                
                # Estimate tokens (rough estimate)
                tokens_used = len(prompt.split()) + len(content.split())
                
                # Log success
                metadata = CallMetadata(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    prompt_hash=prompt_hash,
                    model=model_name,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    success=True
                )
                self._log_call(metadata)
                self.circuit_breaker.record_success()
                
                return GeminiResponse(
                    success=True,
                    content=content,
                    model=model_name,
                    prompt_hash=prompt_hash,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini call failed (attempt {attempt + 1}): {error_msg}")
                
                # Check if we should retry
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    # Final failure
                    self.circuit_breaker.record_failure()
                    latency_ms = (time.time() - start_time) * 1000
                    
                    metadata = CallMetadata(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        prompt_hash=prompt_hash,
                        model=model_name if 'model_name' in locals() else "unknown",
                        tokens_used=0,
                        latency_ms=latency_ms,
                        success=False,
                        error=error_msg
                    )
                    self._log_call(metadata)
                    
                    return GeminiResponse(
                        success=False,
                        content="",
                        model=model_name if 'model_name' in locals() else "unknown",
                        prompt_hash=prompt_hash,
                        tokens_used=0,
                        latency_ms=latency_ms,
                        error=error_msg
                    )
        
        # Should not reach here
        return GeminiResponse(
            success=False,
            content="",
            model="",
            prompt_hash=prompt_hash,
            tokens_used=0,
            latency_ms=0,
            error="Unknown error"
        )
    
    def call_for_json(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.BALANCED,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Gemini and parse response as JSON.
        
        This is useful when we need structured data back from Gemini.
        Like asking someone to fill out a form instead of writing freely.
        """
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanation."
        
        response = self.call(
            prompt=json_prompt,
            tier=tier,
            system_instruction=system_instruction,
            temperature=0.3  # Lower temperature for structured output
        )
        
        if not response.success:
            return {"error": response.error, "success": False}
        
        # Try to parse JSON from response
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_content": response.content,
                "success": False
            }
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about API usage.
        
        Returns stats like total calls, average latency, error rate.
        """
        if not self.call_history:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0,
                "error_rate": 0
            }
        
        successful = [c for c in self.call_history if c.success]
        failed = [c for c in self.call_history if not c.success]
        
        return {
            "total_calls": len(self.call_history),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "total_tokens": sum(c.tokens_used for c in self.call_history),
            "avg_latency_ms": sum(c.latency_ms for c in successful) / len(successful) if successful else 0,
            "error_rate": len(failed) / len(self.call_history) if self.call_history else 0
        }


# Singleton instance for convenience
_default_client: Optional[GeminiClient] = None


def get_client() -> GeminiClient:
    """Get the default Gemini client instance."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client


def call_gemini(
    prompt: str,
    tier: ModelTier = ModelTier.BALANCED,
    system_instruction: Optional[str] = None
) -> GeminiResponse:
    """Convenience function to call Gemini with default client."""
    return get_client().call(prompt, tier, system_instruction)


def call_gemini_json(
    prompt: str,
    tier: ModelTier = ModelTier.BALANCED
) -> Dict[str, Any]:
    """Convenience function to call Gemini and get JSON response."""
    return get_client().call_for_json(prompt, tier)

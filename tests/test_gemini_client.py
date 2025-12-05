"""
Tests for the Gemini Client Module

These tests mock the Gemini API to test:
- Authentication handling
- Retry logic
- Circuit breaker
- Response parsing
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

from gemini_client import (
    GeminiClient, GeminiConfig, GeminiResponse, ModelTier,
    CircuitBreaker, CallMetadata
)


class TestGeminiConfig:
    """Test the configuration class."""
    
    def test_default_config(self):
        """Config should have sensible defaults."""
        config = GeminiConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.fast_model == "gemini-1.5-flash"
        assert config.max_retries == 3
        assert config.timeout == 30
    
    def test_config_from_env(self, monkeypatch):
        """Config should load from environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")
        monkeypatch.setenv("GEMINI_MAX_RETRIES", "5")
        
        config = GeminiConfig.from_env()
        assert config.api_key == "env-test-key"
        assert config.max_retries == 5
    
    def test_config_missing_api_key(self, monkeypatch):
        """Config should warn when API key is missing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = GeminiConfig.from_env()
        assert config.api_key == ""


class TestCircuitBreaker:
    """Test the circuit breaker logic."""
    
    def test_initial_state(self):
        """Circuit breaker should start closed (allowing calls)."""
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.can_proceed() is True
        assert cb.is_open is False
    
    def test_opens_after_threshold(self):
        """Circuit should open after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        assert cb.can_proceed() is True
        
        cb.record_failure()
        assert cb.can_proceed() is True
        
        cb.record_failure()
        assert cb.can_proceed() is False
        assert cb.is_open is True
    
    def test_resets_on_success(self):
        """Circuit should reset after a success."""
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failures == 0
        assert cb.is_open is False
    
    def test_resets_after_timeout(self):
        """Circuit should reset after timeout period."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.can_proceed() is False
        
        # Wait for timeout
        time.sleep(0.15)
        assert cb.can_proceed() is True


class TestGeminiClient:
    """Test the main Gemini client."""
    
    @pytest.fixture
    def mock_genai(self):
        """Create a mock for the google.generativeai module."""
        with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
            yield sys.modules['google.generativeai']
    
    @pytest.fixture
    def client(self):
        """Create a client with test config."""
        config = GeminiConfig(api_key="test-api-key")
        return GeminiClient(config)
    
    def test_hash_prompt(self, client):
        """Should create consistent short hashes."""
        hash1 = client._hash_prompt("test prompt")
        hash2 = client._hash_prompt("test prompt")
        hash3 = client._hash_prompt("different prompt")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 12
    
    def test_calculate_backoff(self, client):
        """Backoff should increase exponentially."""
        delay0 = client._calculate_backoff(0)
        delay1 = client._calculate_backoff(1)
        delay2 = client._calculate_backoff(2)
        
        # Each should be roughly double (with jitter)
        assert delay0 < delay1 < delay2
        assert delay0 < 2  # First delay should be small
    
    def test_backoff_max_limit(self, client):
        """Backoff should not exceed max delay."""
        delay = client._calculate_backoff(100)
        assert delay <= client.config.max_delay * 1.1  # Allow 10% jitter
    
    def test_circuit_breaker_blocks_calls(self, client):
        """Client should not call API when circuit is open."""
        # Force circuit open
        for _ in range(10):
            client.circuit_breaker.record_failure()
        
        response = client.call("test prompt")
        
        assert response.success is False
        assert "Circuit breaker" in response.error
    
    def test_call_success(self, client, mock_genai):
        """Successful API call should return proper response."""
        # Setup mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is the answer"
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        client._genai = mock_genai
        client._models = {}
        
        response = client.call("Test question", tier=ModelTier.FAST)
        
        assert response.success is True
        assert response.content == "This is the answer"
        assert response.error is None
    
    def test_call_logs_metadata(self, client, mock_genai):
        """Calls should log metadata."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer"
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        client._genai = mock_genai
        client._models = {}
        
        client.call("Test")
        
        assert len(client.call_history) == 1
        assert client.call_history[0].success is True
    
    def test_call_for_json_parses_response(self, client, mock_genai):
        """JSON calls should parse the response."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"tldr": "test", "explanation": "details"}'
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        client._genai = mock_genai
        client._models = {}
        
        result = client.call_for_json("Explain something")
        
        assert result["tldr"] == "test"
        assert result["explanation"] == "details"
    
    def test_call_for_json_handles_markdown(self, client, mock_genai):
        """JSON parsing should handle markdown code blocks."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '```json\n{"key": "value"}\n```'
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        client._genai = mock_genai
        client._models = {}
        
        result = client.call_for_json("Test")
        
        assert result["key"] == "value"
    
    def test_analytics(self, client):
        """Analytics should summarize call history."""
        # Add some mock history
        client.call_history = [
            CallMetadata("2024-01-01", "abc", "model", 100, 50, True),
            CallMetadata("2024-01-01", "def", "model", 150, 75, True),
            CallMetadata("2024-01-01", "ghi", "model", 0, 100, False, "error"),
        ]
        
        analytics = client.get_analytics()
        
        assert analytics["total_calls"] == 3
        assert analytics["successful_calls"] == 2
        assert analytics["failed_calls"] == 1
        assert analytics["total_tokens"] == 250
        assert analytics["error_rate"] == pytest.approx(0.333, rel=0.01)


class TestModelTier:
    """Test the model tier enum."""
    
    def test_tier_values(self):
        """Tiers should have expected string values."""
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.BALANCED.value == "balanced"
        assert ModelTier.DEEP.value == "deep"


# Integration-style tests with mocked Gemini

class TestGeminiClientIntegration:
    """Integration tests that mock the full Gemini flow."""
    
    def test_retry_on_failure(self):
        """Client should retry on transient failures."""
        config = GeminiConfig(api_key="test", max_retries=3, base_delay=0.01)
        client = GeminiClient(config)
        
        call_count = 0
        
        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            mock_response = MagicMock()
            mock_response.text = "Success after retries"
            return mock_response
        
        with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
            mock_genai = sys.modules['google.generativeai']
            mock_model = MagicMock()
            mock_model.generate_content = mock_generate
            mock_genai.GenerativeModel.return_value = mock_model
            
            client._genai = mock_genai
            client._models = {}
            
            response = client.call("Test")
            
            assert response.success is True
            assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

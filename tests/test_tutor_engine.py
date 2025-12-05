"""
Tests for the Tutor Engine Module

These tests verify the tutoring logic with mocked Gemini responses.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

from tutor_engine import TutorEngine
from memory_store import MemoryStore
from gemini_client import GeminiClient, GeminiResponse, GeminiConfig


class TestTutorEngineOnboarding:
    """Test onboarding and assessment features."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine(self, mock_gemini, temp_store):
        """Create a tutor engine with mocks."""
        return TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
    
    def test_start_onboarding(self, engine, mock_gemini):
        """Should create learner and generate quiz."""
        # Mock Gemini response
        mock_gemini.call_for_json.return_value = {
            "questions": [
                {
                    "id": 1,
                    "text": "What is 2 + 2?",
                    "correct_answer": "4",
                    "difficulty": "easy",
                    "tests_concept": "addition",
                    "wrong_answer_reveals": "counting error"
                }
            ]
        }
        
        result = engine.start_onboarding(
            topic="Basic Math",
            analogy_domain="cooking",
            num_questions=3
        )
        
        assert "learner_id" in result
        assert "questions" in result
        assert len(result["questions"]) > 0
    
    def test_start_onboarding_fallback(self, engine, mock_gemini):
        """Should use fallback questions if Gemini fails."""
        mock_gemini.call_for_json.return_value = {"error": "API error"}
        
        result = engine.start_onboarding(topic="Math")
        
        assert "learner_id" in result
        assert "questions" in result
        assert len(result["questions"]) >= 1
    
    def test_complete_onboarding(self, engine, mock_gemini, temp_store):
        """Should analyze responses and create profile."""
        # First create a learner
        state = temp_store.create_learner()
        
        # Mock profile analysis
        mock_gemini.call_for_json.return_value = {
            "profile_summary": "Beginner with good potential",
            "knowledge_level": "beginner",
            "misconceptions": [],
            "pace": "medium",
            "confidence": "medium",
            "start_from": "basics"
        }
        
        result = engine.complete_onboarding(
            learner_id=state.learner_id,
            topic="Math",
            responses=[
                {"question": "What is 2+2?", "answer": "4"}
            ]
        )
        
        assert "profile_summary" in result
        assert result["profile"]["knowledge_level"] == "beginner"


class TestTutorEngineExplanations:
    """Test explanation generation."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine_with_learner(self, mock_gemini, temp_store):
        """Create engine with an existing learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        state = temp_store.create_learner(analogy_domain="cooking")
        return engine, state.learner_id
    
    def test_explain_concept(self, engine_with_learner, mock_gemini):
        """Should generate explanation with all parts."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {
            "tldr": "Short summary here",
            "explanation": "Detailed explanation here",
            "analogy": "Think of it like cooking...",
            "worked_example": "Step 1: ...",
            "practice_question": "Try this: ...",
            "hint": "Think about..."
        }
        
        result = engine.explain_concept(
            learner_id=learner_id,
            concept="Pythagorean theorem"
        )
        
        assert result["content"]["tldr"] == "Short summary here"
        assert "analogy" in result["content"]
        assert "practice_question" in result["content"]
    
    def test_explain_concept_unknown_learner(self, mock_gemini, temp_store):
        """Should return error for unknown learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        
        result = engine.explain_concept(
            learner_id="fake-id",
            concept="anything"
        )
        
        assert "error" in result


class TestTutorEngineEvaluation:
    """Test answer evaluation features."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine_with_learner(self, mock_gemini, temp_store):
        """Create engine with an existing learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        state = temp_store.create_learner()
        return engine, state.learner_id
    
    def test_evaluate_correct_answer(self, engine_with_learner, mock_gemini):
        """Should recognize and praise correct answers."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {
            "is_correct": "yes",
            "confidence": 0.95,
            "what_was_right": "Perfect calculation",
            "misconception_detected": None,
            "feedback": "Great job! You got it right.",
            "correction": None,
            "next_suggestion": "Try a harder one"
        }
        
        result = engine.evaluate_answer(
            learner_id=learner_id,
            topic="Math",
            question="What is 5 + 5?",
            expected_answer="10",
            learner_answer="10"
        )
        
        assert result["evaluation"]["is_correct"] == "yes"
        assert result["evaluation"]["feedback"]
    
    def test_evaluate_wrong_answer(self, engine_with_learner, mock_gemini):
        """Should provide correction for wrong answers."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {
            "is_correct": "no",
            "confidence": 0.9,
            "what_was_right": "Good attempt",
            "misconception_detected": "counting error",
            "feedback": "Almost! Let's try again.",
            "correction": "The answer is 10",
            "next_suggestion": "Review addition"
        }
        
        result = engine.evaluate_answer(
            learner_id=learner_id,
            topic="Math",
            question="What is 5 + 5?",
            expected_answer="10",
            learner_answer="11"
        )
        
        assert result["evaluation"]["is_correct"] == "no"
        assert result["evaluation"]["correction"]


class TestTutorEngineHints:
    """Test hint generation."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine_with_learner(self, mock_gemini, temp_store):
        """Create engine with an existing learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        state = temp_store.create_learner()
        return engine, state.learner_id
    
    def test_generate_hint_level_1(self, engine_with_learner, mock_gemini):
        """Should generate subtle hint at level 1."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {
            "hint": "Think about what operation to use",
            "thinking_prompt": "What do you do when combining?",
            "related_concept": "addition basics"
        }
        
        result = engine.generate_hint(
            learner_id=learner_id,
            topic="Math",
            question="What is 3 + 4?",
            wrong_attempt="8",
            hint_level=1
        )
        
        assert "hint" in result["hint"]
        assert result["hint_level"] == 1
    
    def test_generate_hint_fallback(self, engine_with_learner, mock_gemini):
        """Should provide fallback hint on error."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {"error": "API error"}
        
        result = engine.generate_hint(
            learner_id=learner_id,
            topic="Math",
            question="Any question",
            wrong_attempt="wrong",
            hint_level=1
        )
        
        assert "hint" in result
        assert "hint" in result["hint"]  # Fallback hint provided


class TestTutorEngineLessonSummary:
    """Test lesson summary generation."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine_with_learner(self, mock_gemini, temp_store):
        """Create engine with an existing learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        state = temp_store.create_learner()
        return engine, state.learner_id
    
    def test_get_lesson_summary(self, engine_with_learner, mock_gemini):
        """Should generate comprehensive summary."""
        engine, learner_id = engine_with_learner
        
        mock_gemini.call_for_json.return_value = {
            "learned": ["Addition basics", "Number patterns"],
            "unclear": [],
            "practice_items": ["Try 7+8", "Try 9+6"],
            "review_in_days": 2,
            "encouragement": "Great progress today!"
        }
        
        result = engine.get_lesson_summary(
            learner_id=learner_id,
            topic="Addition",
            num_questions=5,
            num_correct=4,
            time_minutes=10.5,
            mistakes=["counted wrong once"]
        )
        
        assert result["score"] == 0.8
        assert len(result["summary"]["learned"]) == 2
        assert result["summary"]["encouragement"]


class TestTutorEngineProgress:
    """Test progress tracking."""
    
    @pytest.fixture
    def mock_gemini(self):
        """Create a mock Gemini client."""
        mock = Mock(spec=GeminiClient)
        return mock
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    @pytest.fixture
    def engine_with_learner(self, mock_gemini, temp_store):
        """Create engine with an existing learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        state = temp_store.create_learner()
        return engine, state.learner_id, temp_store
    
    def test_get_progress(self, engine_with_learner):
        """Should return comprehensive progress."""
        engine, learner_id, store = engine_with_learner
        
        # Add some progress
        store.update_topic_progress(
            learner_id, "math101", "Basic Math", correct=True
        )
        
        result = engine.get_progress(learner_id)
        
        assert result["learner_id"] == learner_id
        assert "profile" in result
        assert "topic_details" in result
    
    def test_get_progress_unknown_learner(self, mock_gemini, temp_store):
        """Should return error for unknown learner."""
        engine = TutorEngine(gemini_client=mock_gemini, memory_store=temp_store)
        
        result = engine.get_progress("fake-id")
        
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

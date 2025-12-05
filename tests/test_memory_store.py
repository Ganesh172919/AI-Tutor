"""
Tests for the Memory Store Module

These tests verify:
- Learner creation and retrieval
- Profile updates
- Progress tracking
- Data export/delete (privacy)
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))

from memory_store import (
    MemoryStore, LearnerState, LearnerProfile, TopicProgress
)


class TestLearnerProfile:
    """Test the LearnerProfile dataclass."""
    
    def test_default_values(self):
        """Profile should have sensible defaults."""
        profile = LearnerProfile(
            learner_id="test123",
            created_at="2024-01-01T10:00:00"
        )
        
        assert profile.knowledge_level == "beginner"
        assert profile.preferred_pace == "medium"
        assert profile.analogy_domain == "everyday"
        assert profile.mode == "simple"
        assert profile.misconceptions == []
    
    def test_custom_values(self):
        """Profile should accept custom values."""
        profile = LearnerProfile(
            learner_id="test123",
            created_at="2024-01-01",
            knowledge_level="advanced",
            analogy_domain="cooking"
        )
        
        assert profile.knowledge_level == "advanced"
        assert profile.analogy_domain == "cooking"


class TestLearnerState:
    """Test the LearnerState dataclass."""
    
    def test_to_dict_and_back(self):
        """State should serialize and deserialize correctly."""
        profile = LearnerProfile(
            learner_id="test123",
            created_at="2024-01-01"
        )
        state = LearnerState(
            learner_id="test123",
            profile=profile,
            mastered_topics=["math101"],
            current_topic="physics"
        )
        
        # Convert to dict and back
        data = state.to_dict()
        restored = LearnerState.from_dict(data)
        
        assert restored.learner_id == state.learner_id
        assert restored.mastered_topics == state.mastered_topics
        assert restored.current_topic == state.current_topic
        assert restored.profile.analogy_domain == profile.analogy_domain
    
    def test_topic_progress_serialization(self):
        """Topic progress should serialize correctly."""
        profile = LearnerProfile(learner_id="test", created_at="2024-01-01")
        state = LearnerState(learner_id="test", profile=profile)
        
        state.topic_progress["math"] = TopicProgress(
            topic_id="math",
            topic_name="Mathematics",
            status="mastered",
            score=0.9,
            attempts=5
        )
        
        data = state.to_dict()
        restored = LearnerState.from_dict(data)
        
        assert "math" in restored.topic_progress
        assert restored.topic_progress["math"].score == 0.9


class TestMemoryStore:
    """Test the MemoryStore class."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    def test_create_learner(self, temp_store):
        """Should create a new learner with unique ID."""
        state = temp_store.create_learner(
            analogy_domain="sports",
            mode="in-depth"
        )
        
        assert state.learner_id is not None
        assert len(state.learner_id) > 10  # UUID format
        assert state.profile.analogy_domain == "sports"
        assert state.profile.mode == "in-depth"
    
    def test_get_learner(self, temp_store):
        """Should retrieve a created learner."""
        created = temp_store.create_learner()
        retrieved = temp_store.get_learner(created.learner_id)
        
        assert retrieved is not None
        assert retrieved.learner_id == created.learner_id
    
    def test_get_nonexistent_learner(self, temp_store):
        """Should return None for nonexistent learner."""
        result = temp_store.get_learner("fake-id-12345")
        assert result is None
    
    def test_save_learner(self, temp_store):
        """Should persist learner changes."""
        state = temp_store.create_learner()
        state.mastered_topics.append("test_topic")
        state.session_count = 5
        
        temp_store.save_learner(state)
        
        # Retrieve and verify
        retrieved = temp_store.get_learner(state.learner_id)
        assert "test_topic" in retrieved.mastered_topics
        assert retrieved.session_count == 5
    
    def test_delete_learner(self, temp_store):
        """Should delete learner data."""
        state = temp_store.create_learner()
        learner_id = state.learner_id
        
        # Verify exists
        assert temp_store.get_learner(learner_id) is not None
        
        # Delete
        success = temp_store.delete_learner(learner_id)
        assert success is True
        
        # Verify gone
        assert temp_store.get_learner(learner_id) is None
    
    def test_delete_nonexistent(self, temp_store):
        """Should return False for deleting nonexistent learner."""
        success = temp_store.delete_learner("fake-id")
        assert success is False
    
    def test_export_learner(self, temp_store):
        """Should export learner data as dict."""
        state = temp_store.create_learner()
        state.mastered_topics = ["topic1", "topic2"]
        temp_store.save_learner(state)
        
        exported = temp_store.export_learner(state.learner_id)
        
        assert exported is not None
        assert exported["learner_id"] == state.learner_id
        assert exported["mastered_topics"] == ["topic1", "topic2"]
    
    def test_import_learner(self, temp_store):
        """Should import learner data from dict."""
        data = {
            "learner_id": "imported-123",
            "profile": {
                "learner_id": "imported-123",
                "created_at": "2024-01-01",
                "knowledge_level": "intermediate",
                "preferred_pace": "fast",
                "confidence": "high",
                "analogy_domain": "cooking",
                "mode": "simple",
                "misconceptions": [],
                "interests": []
            },
            "mastered_topics": ["imported_topic"],
            "pending_topics": [],
            "topic_progress": {}
        }
        
        state = temp_store.import_learner(data)
        
        assert state is not None
        assert state.learner_id == "imported-123"
        assert "imported_topic" in state.mastered_topics
        
        # Verify persisted
        retrieved = temp_store.get_learner("imported-123")
        assert retrieved is not None
    
    def test_update_profile(self, temp_store):
        """Should update specific profile fields."""
        state = temp_store.create_learner()
        
        temp_store.update_profile(
            state.learner_id,
            analogy_domain="money",
            knowledge_level="advanced"
        )
        
        updated = temp_store.get_learner(state.learner_id)
        assert updated.profile.analogy_domain == "money"
        assert updated.profile.knowledge_level == "advanced"


class TestTopicProgress:
    """Test topic progress tracking."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    def test_update_progress_correct(self, temp_store):
        """Should track correct answers."""
        state = temp_store.create_learner()
        
        temp_store.update_topic_progress(
            state.learner_id,
            topic_id="math101",
            topic_name="Basic Math",
            correct=True
        )
        
        updated = temp_store.get_learner(state.learner_id)
        progress = updated.topic_progress["math101"]
        
        assert progress.attempts == 1
        assert progress.correct_answers == 1
        assert progress.score == 1.0
    
    def test_update_progress_incorrect(self, temp_store):
        """Should track incorrect answers and mistakes."""
        state = temp_store.create_learner()
        
        temp_store.update_topic_progress(
            state.learner_id,
            topic_id="math101",
            topic_name="Basic Math",
            correct=False,
            mistake="forgot to carry"
        )
        
        updated = temp_store.get_learner(state.learner_id)
        progress = updated.topic_progress["math101"]
        
        assert progress.attempts == 1
        assert progress.correct_answers == 0
        assert "forgot to carry" in progress.common_mistakes
    
    def test_mastery_after_multiple_correct(self, temp_store):
        """Should mark topic as mastered after good performance."""
        state = temp_store.create_learner()
        
        # Answer correctly 4 out of 5 times
        for i in range(5):
            temp_store.update_topic_progress(
                state.learner_id,
                topic_id="math101",
                topic_name="Basic Math",
                correct=(i != 2)  # One wrong
            )
        
        updated = temp_store.get_learner(state.learner_id)
        progress = updated.topic_progress["math101"]
        
        assert progress.status == "mastered"
        assert progress.score == 0.8  # 4/5
        assert "math101" in updated.mastered_topics


class TestAnalytics:
    """Test analytics features."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            yield store
    
    def test_analytics_summary_empty(self, temp_store):
        """Should return zeros for empty store."""
        summary = temp_store.get_analytics_summary()
        
        assert summary["total_learners"] == 0
        assert summary["total_learning_hours"] == 0
    
    def test_session_tracking(self, temp_store):
        """Should track session time."""
        state = temp_store.create_learner()
        
        temp_store.start_session(state.learner_id)
        temp_store.end_session(state.learner_id, duration_minutes=15.5)
        
        updated = temp_store.get_learner(state.learner_id)
        assert updated.session_count == 1
        assert updated.total_time_minutes == 15.5
    
    def test_list_learners(self, temp_store):
        """Should list all learner IDs."""
        # Create some learners
        state1 = temp_store.create_learner()
        state2 = temp_store.create_learner()
        
        learners = temp_store.list_learners()
        
        assert state1.learner_id in learners
        assert state2.learner_id in learners


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

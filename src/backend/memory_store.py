"""
Memory Store Module

This module handles storing and retrieving learner data.
Think of it like a notebook where we keep track of each student's progress.

Default: JSON file storage (good for development)
Production: Can swap to PostgreSQL or Redis (instructions in comments)
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import uuid
import hashlib


# Default storage directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class LearnerProfile:
    """
    A learner's profile and preferences.
    
    This stores who they are and how they like to learn.
    """
    learner_id: str
    created_at: str
    knowledge_level: str = "beginner"  # beginner, intermediate, advanced
    preferred_pace: str = "medium"     # slow, medium, fast
    confidence: str = "medium"         # low, medium, high
    analogy_domain: str = "everyday"   # cooking, sports, everyday, etc.
    mode: str = "simple"               # simple (ELI5) or in-depth
    misconceptions: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)


@dataclass 
class TopicProgress:
    """
    Progress on a specific topic.
    
    Like a report card for one subject.
    """
    topic_id: str
    topic_name: str
    status: str = "not_started"  # not_started, in_progress, mastered
    score: float = 0.0           # 0 to 1
    attempts: int = 0
    correct_answers: int = 0
    common_mistakes: List[str] = field(default_factory=list)
    last_practiced: Optional[str] = None
    next_review: Optional[str] = None  # For spaced repetition


@dataclass
class LearnerState:
    """
    Complete state for a learner.
    
    This is everything we remember about a student:
    - Who they are (profile)
    - What they know (mastered topics)
    - What they're learning (pending topics)  
    - Where they struggle (mistakes)
    """
    learner_id: str
    profile: LearnerProfile
    mastered_topics: List[str] = field(default_factory=list)
    pending_topics: List[str] = field(default_factory=list)
    current_topic: Optional[str] = None
    last_difficulty: str = "easy"
    topic_progress: Dict[str, TopicProgress] = field(default_factory=dict)
    session_count: int = 0
    total_time_minutes: float = 0.0
    last_session: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return {
            "learner_id": self.learner_id,
            "profile": asdict(self.profile),
            "mastered_topics": self.mastered_topics,
            "pending_topics": self.pending_topics,
            "current_topic": self.current_topic,
            "last_difficulty": self.last_difficulty,
            "topic_progress": {
                k: asdict(v) for k, v in self.topic_progress.items()
            },
            "session_count": self.session_count,
            "total_time_minutes": self.total_time_minutes,
            "last_session": self.last_session
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LearnerState":
        """Create from dictionary."""
        profile_data = data.get("profile", {})
        profile = LearnerProfile(**profile_data)
        
        topic_progress = {}
        for k, v in data.get("topic_progress", {}).items():
            topic_progress[k] = TopicProgress(**v)
        
        return cls(
            learner_id=data["learner_id"],
            profile=profile,
            mastered_topics=data.get("mastered_topics", []),
            pending_topics=data.get("pending_topics", []),
            current_topic=data.get("current_topic"),
            last_difficulty=data.get("last_difficulty", "easy"),
            topic_progress=topic_progress,
            session_count=data.get("session_count", 0),
            total_time_minutes=data.get("total_time_minutes", 0.0),
            last_session=data.get("last_session")
        )


@dataclass
class AnalyticsData:
    """
    Analytics about a learner's usage.
    
    This is separate from PII - just numbers and patterns.
    """
    learner_hash: str  # Anonymized ID
    lessons_completed: int = 0
    average_score: float = 0.0
    total_time_minutes: float = 0.0
    topics_mastered: int = 0
    common_mistake_patterns: List[str] = field(default_factory=list)
    retention_rate: float = 0.0


class MemoryStore:
    """
    Main storage class for learner data.
    
    Uses JSON files by default. Each learner has their own file.
    This makes it easy to:
    - Export a learner's data
    - Delete a learner's data
    - Back up everything
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the store with a data directory."""
        self.data_dir = data_dir or DATA_DIR
        self.learners_dir = self.data_dir / "learners"
        self.analytics_dir = self.data_dir / "analytics"
        
        # Create directories if they don't exist
        os.makedirs(self.learners_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
    
    def _get_learner_path(self, learner_id: str) -> Path:
        """Get the file path for a learner's data."""
        # Sanitize ID to prevent path traversal
        safe_id = "".join(c for c in learner_id if c.isalnum() or c in "-_")
        return self.learners_dir / f"{safe_id}.json"
    
    def _hash_learner_id(self, learner_id: str) -> str:
        """Create an anonymized hash of learner ID for analytics."""
        return hashlib.sha256(learner_id.encode()).hexdigest()[:16]
    
    def create_learner(
        self,
        analogy_domain: str = "everyday",
        mode: str = "simple"
    ) -> LearnerState:
        """
        Create a new learner with a fresh profile.
        
        Args:
            analogy_domain: What kind of analogies they prefer
            mode: simple (ELI5) or in-depth
            
        Returns:
            A new LearnerState with unique ID
        """
        learner_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        profile = LearnerProfile(
            learner_id=learner_id,
            created_at=now,
            analogy_domain=analogy_domain,
            mode=mode
        )
        
        state = LearnerState(
            learner_id=learner_id,
            profile=profile,
            last_session=now
        )
        
        # Save immediately
        self.save_learner(state)
        
        return state
    
    def get_learner(self, learner_id: str) -> Optional[LearnerState]:
        """
        Get a learner's state by ID.
        
        Returns None if not found.
        """
        path = self._get_learner_path(learner_id)
        if not path.exists():
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return LearnerState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading learner {learner_id}: {e}")
            return None
    
    def save_learner(self, state: LearnerState) -> bool:
        """
        Save a learner's state to disk.
        
        Returns True if successful.
        """
        path = self._get_learner_path(state.learner_id)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving learner {state.learner_id}: {e}")
            return False
    
    def delete_learner(self, learner_id: str) -> bool:
        """
        Delete all data for a learner.
        
        This is for privacy compliance - users can request deletion.
        """
        path = self._get_learner_path(learner_id)
        if path.exists():
            try:
                os.remove(path)
                return True
            except Exception as e:
                print(f"Error deleting learner {learner_id}: {e}")
                return False
        return False
    
    def export_learner(self, learner_id: str) -> Optional[Dict]:
        """
        Export all data for a learner as JSON.
        
        This is for data portability - users can download their data.
        """
        state = self.get_learner(learner_id)
        if state:
            return state.to_dict()
        return None
    
    def import_learner(self, data: Dict) -> Optional[LearnerState]:
        """
        Import learner data from JSON.
        
        Returns the imported state or None if invalid.
        """
        try:
            state = LearnerState.from_dict(data)
            self.save_learner(state)
            return state
        except Exception as e:
            print(f"Error importing learner: {e}")
            return None
    
    def update_profile(
        self,
        learner_id: str,
        **updates
    ) -> Optional[LearnerState]:
        """
        Update a learner's profile.
        
        Pass any profile fields as keyword arguments.
        """
        state = self.get_learner(learner_id)
        if not state:
            return None
        
        for key, value in updates.items():
            if hasattr(state.profile, key):
                setattr(state.profile, key, value)
        
        self.save_learner(state)
        return state
    
    def update_topic_progress(
        self,
        learner_id: str,
        topic_id: str,
        topic_name: str,
        correct: bool,
        mistake: Optional[str] = None
    ) -> Optional[LearnerState]:
        """
        Update progress on a topic after an answer.
        
        Args:
            learner_id: The learner's ID
            topic_id: Unique ID for the topic
            topic_name: Human-readable name
            correct: Whether the answer was correct
            mistake: Description of mistake (if wrong)
        """
        state = self.get_learner(learner_id)
        if not state:
            return None
        
        # Get or create topic progress
        if topic_id not in state.topic_progress:
            state.topic_progress[topic_id] = TopicProgress(
                topic_id=topic_id,
                topic_name=topic_name
            )
        
        progress = state.topic_progress[topic_id]
        progress.attempts += 1
        progress.last_practiced = datetime.now().isoformat()
        
        if correct:
            progress.correct_answers += 1
        elif mistake and mistake not in progress.common_mistakes:
            progress.common_mistakes.append(mistake)
        
        # Calculate score
        if progress.attempts > 0:
            progress.score = progress.correct_answers / progress.attempts
        
        # Update status based on score
        if progress.score >= 0.8 and progress.attempts >= 3:
            progress.status = "mastered"
            if topic_id not in state.mastered_topics:
                state.mastered_topics.append(topic_id)
            if topic_id in state.pending_topics:
                state.pending_topics.remove(topic_id)
        elif progress.attempts > 0:
            progress.status = "in_progress"
            if topic_id not in state.pending_topics and topic_id not in state.mastered_topics:
                state.pending_topics.append(topic_id)
        
        # Set spaced repetition review date
        if progress.status == "mastered":
            # Review in 1, 3, 7, 14 days based on how well they know it
            days = 1 if progress.score < 0.9 else 3
            progress.next_review = (
                datetime.now().isoformat()
            )
        
        self.save_learner(state)
        return state
    
    def get_topics_for_review(self, learner_id: str) -> List[TopicProgress]:
        """
        Get topics that are due for spaced repetition review.
        
        Returns topics where next_review date has passed.
        """
        state = self.get_learner(learner_id)
        if not state:
            return []
        
        now = datetime.now().isoformat()
        due_topics = []
        
        for progress in state.topic_progress.values():
            if progress.next_review and progress.next_review <= now:
                due_topics.append(progress)
        
        return due_topics
    
    def start_session(self, learner_id: str) -> Optional[LearnerState]:
        """Mark the start of a learning session."""
        state = self.get_learner(learner_id)
        if not state:
            return None
        
        state.session_count += 1
        state.last_session = datetime.now().isoformat()
        self.save_learner(state)
        return state
    
    def end_session(
        self,
        learner_id: str,
        duration_minutes: float
    ) -> Optional[LearnerState]:
        """Mark the end of a session and record time."""
        state = self.get_learner(learner_id)
        if not state:
            return None
        
        state.total_time_minutes += duration_minutes
        self.save_learner(state)
        
        # Also update analytics (anonymized)
        self._update_analytics(learner_id, duration_minutes)
        
        return state
    
    def _update_analytics(self, learner_id: str, duration_minutes: float):
        """Update anonymized analytics."""
        learner_hash = self._hash_learner_id(learner_id)
        analytics_path = self.analytics_dir / f"{learner_hash}.json"
        
        # Load existing or create new
        if analytics_path.exists():
            with open(analytics_path, 'r') as f:
                data = json.load(f)
            analytics = AnalyticsData(**data)
        else:
            analytics = AnalyticsData(learner_hash=learner_hash)
        
        # Update
        analytics.total_time_minutes += duration_minutes
        
        # Save
        with open(analytics_path, 'w') as f:
            json.dump(asdict(analytics), f, indent=2)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get aggregate analytics across all learners.
        
        This is anonymized - no individual data exposed.
        """
        total_learners = 0
        total_time = 0.0
        total_topics_mastered = 0
        
        for file_path in self.analytics_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                total_learners += 1
                total_time += data.get("total_time_minutes", 0)
                total_topics_mastered += data.get("topics_mastered", 0)
            except:
                continue
        
        return {
            "total_learners": total_learners,
            "total_learning_hours": total_time / 60,
            "total_topics_mastered": total_topics_mastered,
            "avg_topics_per_learner": (
                total_topics_mastered / total_learners if total_learners > 0 else 0
            )
        }
    
    def list_learners(self) -> List[str]:
        """List all learner IDs (for admin purposes)."""
        learner_ids = []
        for file_path in self.learners_dir.glob("*.json"):
            learner_ids.append(file_path.stem)
        return learner_ids


# Default store instance
_default_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    """Get the default memory store instance."""
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


# Convenience functions

def create_learner(**kwargs) -> LearnerState:
    """Create a new learner."""
    return get_store().create_learner(**kwargs)


def get_learner(learner_id: str) -> Optional[LearnerState]:
    """Get a learner by ID."""
    return get_store().get_learner(learner_id)


def save_learner(state: LearnerState) -> bool:
    """Save learner state."""
    return get_store().save_learner(state)


def delete_learner(learner_id: str) -> bool:
    """Delete a learner."""
    return get_store().delete_learner(learner_id)


def export_learner(learner_id: str) -> Optional[Dict]:
    """Export learner data."""
    return get_store().export_learner(learner_id)


"""
Production Storage Options
==========================

To use PostgreSQL instead of JSON files:

1. Install: pip install asyncpg sqlalchemy

2. Replace the MemoryStore class with PostgreSQL implementation:

```python
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/aitutor")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class LearnerModel(Base):
    __tablename__ = "learners"
    learner_id = Column(String, primary_key=True)
    data = Column(JSON)
```

To use Redis for session management:

1. Install: pip install redis

2. Use Redis for active session state:

```python
import redis

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost"))

def cache_session(learner_id: str, state: dict, ttl: int = 3600):
    redis_client.setex(f"session:{learner_id}", ttl, json.dumps(state))

def get_cached_session(learner_id: str) -> Optional[dict]:
    data = redis_client.get(f"session:{learner_id}")
    return json.loads(data) if data else None
```
"""

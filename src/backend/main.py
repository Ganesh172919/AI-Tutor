"""
AI Tutor FastAPI Application

This is the main REST API for the AI Tutor system.
It provides endpoints for:
- Onboarding and assessment
- Lesson planning
- Explanations and chat
- Progress tracking
- Data export/delete (privacy)
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .gemini_client import GeminiClient, get_client
from .memory_store import MemoryStore, get_store, LearnerState
from .tutor_engine import TutorEngine, get_engine


# === Pydantic Models for Request/Response ===

class StartOnboardingRequest(BaseModel):
    """Request to start onboarding."""
    topic: str = Field(..., description="Subject to learn")
    analogy_domain: str = Field(
        default="everyday",
        description="Preferred analogy type: cooking, sports, everyday, etc."
    )
    mode: str = Field(
        default="simple",
        description="Explanation mode: simple (ELI5) or in-depth"
    )
    num_questions: int = Field(
        default=5,
        ge=3,
        le=7,
        description="Number of assessment questions"
    )


class QuizResponse(BaseModel):
    """A single quiz answer."""
    question: str
    answer: str


class CompleteOnboardingRequest(BaseModel):
    """Request to complete onboarding with quiz answers."""
    learner_id: str
    topic: str
    responses: List[QuizResponse]


class CreateLessonPlanRequest(BaseModel):
    """Request to create a lesson plan."""
    learner_id: str
    topic: str


class ExplainConceptRequest(BaseModel):
    """Request to explain a concept."""
    learner_id: str
    concept: str


class EvaluateAnswerRequest(BaseModel):
    """Request to evaluate an answer."""
    learner_id: str
    topic: str
    question: str
    expected_answer: str
    learner_answer: str


class GenerateHintRequest(BaseModel):
    """Request to generate a hint."""
    learner_id: str
    topic: str
    question: str
    wrong_attempt: str
    hint_level: int = Field(default=1, ge=1, le=3)


class LessonSummaryRequest(BaseModel):
    """Request for lesson summary."""
    learner_id: str
    topic: str
    num_questions: int
    num_correct: int
    time_minutes: float
    mistakes: List[str] = []


class UpdateProfileRequest(BaseModel):
    """Request to update learner profile."""
    analogy_domain: Optional[str] = None
    mode: Optional[str] = None
    knowledge_level: Optional[str] = None
    preferred_pace: Optional[str] = None


class ChatMessage(BaseModel):
    """A chat message in the tutoring session."""
    learner_id: str
    message: str
    context_topic: Optional[str] = None


# === Session Tracking ===

class SessionTracker:
    """Track active learning sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def start_session(self, learner_id: str) -> str:
        """Start a new session."""
        session_id = f"{learner_id}_{int(time.time())}"
        self.sessions[session_id] = {
            "learner_id": learner_id,
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "questions_attempted": 0,
            "correct_answers": 0
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, **updates):
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)


session_tracker = SessionTracker()


# === App Lifecycle ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    print("Starting AI Tutor API...")
    print(f"Gemini API Key configured: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    yield
    # Shutdown
    print("Shutting down AI Tutor API...")


# === Create FastAPI App ===

app = FastAPI(
    title="AI Tutor API",
    description="An AI-powered tutor that teaches any subject with simple explanations and real-life examples.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Dependency Injection ===

def get_tutor_engine() -> TutorEngine:
    """Get the tutor engine."""
    return get_engine()


def get_memory_store() -> MemoryStore:
    """Get the memory store."""
    return get_store()


# === Health Check ===

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


# === Onboarding Endpoints ===

@app.post("/api/onboarding/start")
async def start_onboarding(
    request: StartOnboardingRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Start the onboarding process for a new learner.
    
    This creates a learner profile and generates assessment questions.
    """
    result = engine.start_onboarding(
        topic=request.topic,
        analogy_domain=request.analogy_domain,
        mode=request.mode,
        num_questions=request.num_questions
    )
    
    if "error" in result and "learner_id" not in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result


@app.post("/api/onboarding/complete")
async def complete_onboarding(
    request: CompleteOnboardingRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Complete onboarding by analyzing quiz responses.
    
    Returns a learner profile with knowledge level and recommendations.
    """
    responses = [{"question": r.question, "answer": r.answer} for r in request.responses]
    
    result = engine.complete_onboarding(
        learner_id=request.learner_id,
        topic=request.topic,
        responses=responses
    )
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# === Lesson Endpoints ===

@app.post("/api/lesson/plan")
async def create_lesson_plan(
    request: CreateLessonPlanRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Create a micro-lesson plan for a topic.
    
    Breaks the topic into small, digestible chunks.
    """
    result = engine.create_lesson_plan(
        learner_id=request.learner_id,
        topic=request.topic
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.post("/api/lesson/explain")
async def explain_concept(
    request: ExplainConceptRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Explain a concept with simple language and real-life examples.
    
    Returns TL;DR, explanation, analogy, worked example, and practice question.
    """
    result = engine.explain_concept(
        learner_id=request.learner_id,
        concept=request.concept
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


# === Evaluation Endpoints ===

@app.post("/api/evaluate/answer")
async def evaluate_answer(
    request: EvaluateAnswerRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Evaluate a learner's answer and provide feedback.
    
    Feedback is encouraging and includes corrections if needed.
    """
    result = engine.evaluate_answer(
        learner_id=request.learner_id,
        topic=request.topic,
        question=request.question,
        expected_answer=request.expected_answer,
        learner_answer=request.learner_answer
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.post("/api/evaluate/hint")
async def generate_hint(
    request: GenerateHintRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Generate a hint for a struggling learner.
    
    Hints get more direct as hint_level increases (1-3).
    """
    result = engine.generate_hint(
        learner_id=request.learner_id,
        topic=request.topic,
        question=request.question,
        wrong_attempt=request.wrong_attempt,
        hint_level=request.hint_level
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.post("/api/evaluate/summary")
async def get_lesson_summary(
    request: LessonSummaryRequest,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Get a summary after completing a micro-lesson.
    
    Returns what was learned, unclear points, and practice suggestions.
    """
    result = engine.get_lesson_summary(
        learner_id=request.learner_id,
        topic=request.topic,
        num_questions=request.num_questions,
        num_correct=request.num_correct,
        time_minutes=request.time_minutes,
        mistakes=request.mistakes
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


# === Progress Endpoints ===

@app.get("/api/progress/{learner_id}")
async def get_progress(
    learner_id: str,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """Get overall progress for a learner."""
    result = engine.get_progress(learner_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@app.get("/api/progress/{learner_id}/review")
async def get_review_topics(
    learner_id: str,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """Get topics due for spaced repetition review."""
    return engine.get_topics_for_review(learner_id)


@app.put("/api/profile/{learner_id}")
async def update_profile(
    learner_id: str,
    request: UpdateProfileRequest,
    store: MemoryStore = Depends(get_memory_store)
):
    """Update a learner's profile settings."""
    updates = {k: v for k, v in request.dict().items() if v is not None}
    
    result = store.update_profile(learner_id, **updates)
    
    if not result:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    return {"status": "updated", "learner_id": learner_id}


# === Data Privacy Endpoints ===

@app.get("/api/data/{learner_id}/export")
async def export_data(
    learner_id: str,
    store: MemoryStore = Depends(get_memory_store)
):
    """
    Export all data for a learner.
    
    Returns a JSON file with all learner data for data portability.
    """
    data = store.export_learner(learner_id)
    
    if not data:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    return {
        "learner_id": learner_id,
        "exported_at": datetime.now().isoformat(),
        "data": data
    }


@app.delete("/api/data/{learner_id}")
async def delete_data(
    learner_id: str,
    store: MemoryStore = Depends(get_memory_store)
):
    """
    Delete all data for a learner.
    
    This is for privacy compliance - users can request deletion.
    """
    success = store.delete_learner(learner_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    return {
        "status": "deleted",
        "learner_id": learner_id,
        "deleted_at": datetime.now().isoformat()
    }


# === Chat Endpoint ===

@app.post("/api/chat")
async def chat(
    message: ChatMessage,
    engine: TutorEngine = Depends(get_tutor_engine)
):
    """
    Have a conversational exchange with the tutor.
    
    This is for free-form questions within a topic.
    """
    # Get learner state
    state = get_store().get_learner(message.learner_id)
    if not state:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    # Determine topic context
    topic = message.context_topic or state.current_topic or "general"
    
    # Build chat prompt
    chat_prompt = f"""You are a helpful AI tutor. Answer this question simply.

Learner's level: {state.profile.knowledge_level}
Preferred analogies: {state.profile.analogy_domain}
Topic context: {topic}

Question: {message.message}

Rules:
1. Use simple English
2. Include one real-life example
3. Keep it under 150 words
4. If it's a question, also give a practice problem

Response:"""
    
    response = get_client().call(
        prompt=chat_prompt,
        system_instruction=f"You teach using {state.profile.analogy_domain} analogies."
    )
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.error)
    
    return {
        "learner_id": message.learner_id,
        "response": response.content,
        "topic": topic
    }


# === Analytics Endpoints ===

@app.get("/api/analytics/summary")
async def get_analytics_summary(
    store: MemoryStore = Depends(get_memory_store)
):
    """
    Get aggregate analytics across all learners.
    
    This is anonymized data for system monitoring.
    """
    return store.get_analytics_summary()


@app.get("/api/analytics/gemini")
async def get_gemini_analytics():
    """Get Gemini API usage statistics."""
    return get_client().get_analytics()


# === Session Endpoints ===

@app.post("/api/session/start/{learner_id}")
async def start_session(
    learner_id: str,
    store: MemoryStore = Depends(get_memory_store)
):
    """Start a new learning session."""
    state = store.start_session(learner_id)
    if not state:
        raise HTTPException(status_code=404, detail="Learner not found")
    
    session_id = session_tracker.start_session(learner_id)
    
    return {
        "session_id": session_id,
        "learner_id": learner_id,
        "started_at": datetime.now().isoformat()
    }


@app.post("/api/session/end")
async def end_session(
    session_id: str,
    store: MemoryStore = Depends(get_memory_store)
):
    """End a learning session and record time."""
    session = session_tracker.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate duration
    started = datetime.fromisoformat(session["started_at"])
    duration = (datetime.now() - started).total_seconds() / 60
    
    # Update learner
    store.end_session(session["learner_id"], duration)
    
    return {
        "session_id": session_id,
        "duration_minutes": round(duration, 2),
        "questions_attempted": session.get("questions_attempted", 0),
        "correct_answers": session.get("correct_answers", 0)
    }


# === Error Handler ===

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc),
            "tip": "Please try again. If the problem persists, check your API key."
        }
    )


# === Run Configuration ===

def create_app() -> FastAPI:
    """Factory function to create the app."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

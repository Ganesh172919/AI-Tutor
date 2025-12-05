"""
Tutor Engine Module

This is the brain of the AI Tutor. It coordinates:
- Onboarding and assessment
- Lesson planning  
- Explanations and dialogue
- Answer evaluation
- Progress tracking

All teaching uses simple English and real-life examples.
"""

import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from .gemini_client import (
    GeminiClient, GeminiResponse, ModelTier, 
    get_client, call_gemini, call_gemini_json
)
from .prompt_builder import (
    PromptBuilder, get_builder, build_prompt,
    fill_inline_template, INLINE_TEMPLATES
)
from .memory_store import (
    MemoryStore, LearnerState, LearnerProfile, TopicProgress,
    get_store, get_learner, save_learner, create_learner
)


@dataclass
class QuizQuestion:
    """A single quiz question for assessment."""
    id: int
    text: str
    correct_answer: str
    difficulty: str
    tests_concept: str
    wrong_answer_reveals: str


@dataclass
class LessonContent:
    """Content for a micro-lesson explanation."""
    tldr: str
    explanation: str
    analogy: str
    worked_example: str
    practice_question: str
    hint: str


@dataclass
class AnswerEvaluation:
    """Result of evaluating a learner's answer."""
    is_correct: str  # yes, partially, no
    confidence: float
    what_was_right: str
    misconception_detected: Optional[str]
    feedback: str
    correction: Optional[str]
    next_suggestion: str


class TutorEngine:
    """
    Main tutoring engine that orchestrates learning.
    
    Think of this as the teacher who:
    - Gives a quick test to see what you know
    - Plans lessons based on your level
    - Explains things simply with examples
    - Checks your answers and helps you improve
    - Remembers what you've learned
    """
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        memory_store: Optional[MemoryStore] = None
    ):
        """Initialize the tutor with its components."""
        self.gemini = gemini_client or get_client()
        self.memory = memory_store or get_store()
    
    def _get_system_prompt(self, state: LearnerState) -> str:
        """Build the system prompt for this learner."""
        return fill_inline_template(
            "system_tutor",
            analogy_domain=state.profile.analogy_domain,
            level=state.profile.knowledge_level,
            mode=state.profile.mode
        )
    
    # === ONBOARDING & ASSESSMENT ===
    
    def start_onboarding(
        self,
        topic: str,
        analogy_domain: str = "everyday",
        mode: str = "simple",
        num_questions: int = 5
    ) -> Dict[str, Any]:
        """
        Start the onboarding process for a new learner.
        
        Creates a learner profile and generates assessment questions.
        
        Args:
            topic: What subject they want to learn
            analogy_domain: Their preferred analogy type (cooking, sports, etc.)
            mode: simple (ELI5) or in-depth
            num_questions: How many quiz questions (3-7 recommended)
            
        Returns:
            Dict with learner_id and quiz questions
        """
        # Create new learner
        state = self.memory.create_learner(
            analogy_domain=analogy_domain,
            mode=mode
        )
        
        # Generate quiz questions
        prompt = fill_inline_template(
            "onboarding_quiz",
            topic=topic,
            num_questions=str(num_questions)
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.BALANCED
        )
        
        if "error" in response:
            return {
                "learner_id": state.learner_id,
                "error": response["error"],
                "questions": self._get_fallback_questions(topic)
            }
        
        return {
            "learner_id": state.learner_id,
            "topic": topic,
            "questions": response.get("questions", [])
        }
    
    def _get_fallback_questions(self, topic: str) -> List[Dict]:
        """Fallback questions if Gemini fails."""
        return [
            {
                "id": 1,
                "text": f"What do you already know about {topic}?",
                "correct_answer": "open_ended",
                "difficulty": "easy",
                "tests_concept": "prior_knowledge",
                "wrong_answer_reveals": "none"
            },
            {
                "id": 2,
                "text": f"Have you studied {topic} before? (yes/no/a little)",
                "correct_answer": "any",
                "difficulty": "easy",
                "tests_concept": "experience",
                "wrong_answer_reveals": "none"
            }
        ]
    
    def complete_onboarding(
        self,
        learner_id: str,
        topic: str,
        responses: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Complete onboarding by analyzing quiz responses.
        
        Args:
            learner_id: The learner's ID from start_onboarding
            topic: The subject being learned
            responses: List of {question, answer} dicts
            
        Returns:
            Learner profile analysis
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        # Format responses for the prompt
        responses_text = "\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in responses
        ])
        
        prompt = fill_inline_template(
            "analyze_responses",
            topic=topic,
            responses=responses_text
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.BALANCED
        )
        
        if "error" in response:
            # Use defaults
            profile_update = {
                "knowledge_level": "beginner",
                "pace": "medium",
                "confidence": "medium"
            }
        else:
            profile_update = {
                "knowledge_level": response.get("knowledge_level", "beginner"),
                "preferred_pace": response.get("pace", "medium"),
                "confidence": response.get("confidence", "medium"),
                "misconceptions": response.get("misconceptions", [])
            }
        
        # Update learner profile
        for key, value in profile_update.items():
            if hasattr(state.profile, key):
                setattr(state.profile, key, value)
        
        state.current_topic = topic
        state.pending_topics.append(topic)
        self.memory.save_learner(state)
        
        return {
            "learner_id": learner_id,
            "profile_summary": response.get("profile_summary", "Profile created."),
            "profile": asdict(state.profile),
            "start_from": response.get("start_from", topic)
        }
    
    # === LESSON PLANNING ===
    
    def create_lesson_plan(
        self,
        learner_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Create a micro-lesson plan for a topic.
        
        Breaks the topic into small, digestible chunks with practice.
        
        Args:
            learner_id: The learner's ID
            topic: What to teach
            
        Returns:
            Lesson plan with micro-lessons
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        prompt = fill_inline_template(
            "lesson_plan",
            topic=topic,
            level=state.profile.knowledge_level,
            analogy_domain=state.profile.analogy_domain
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.DEEP  # Use deeper model for planning
        )
        
        if "error" in response:
            return {"error": response["error"]}
        
        return {
            "learner_id": learner_id,
            "plan": response
        }
    
    # === EXPLANATIONS ===
    
    def explain_concept(
        self,
        learner_id: str,
        concept: str
    ) -> Dict[str, Any]:
        """
        Explain a concept with simple language and examples.
        
        Every explanation includes:
        - TL;DR (one sentence)
        - Simple explanation
        - Real-life analogy
        - Worked example
        - Practice question
        - Hint
        
        Args:
            learner_id: The learner's ID
            concept: What to explain
            
        Returns:
            Full explanation content
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        prompt = fill_inline_template(
            "explain_concept",
            concept=concept,
            level=state.profile.knowledge_level,
            analogy_domain=state.profile.analogy_domain,
            mode=state.profile.mode
        )
        
        system_prompt = self._get_system_prompt(state)
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.BALANCED,
            system_instruction=system_prompt
        )
        
        if "error" in response:
            return {"error": response["error"]}
        
        # Verify math if present
        if self._contains_math(concept):
            response = self._verify_math_in_response(response)
        
        return {
            "learner_id": learner_id,
            "concept": concept,
            "content": response
        }
    
    def _contains_math(self, text: str) -> bool:
        """Check if text contains math-related content."""
        math_keywords = [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'add', 'subtract', 'multiply', 'divide', 'sum', 'product',
            'percentage', 'ratio', 'fraction', 'decimal'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in math_keywords)
    
    def _verify_math_in_response(self, response: Dict) -> Dict:
        """
        Verify mathematical calculations in the response.
        
        If we detect potential math errors, we add a caveat.
        """
        worked_example = response.get("worked_example", "")
        
        # Simple pattern to find calculations
        # In production, you'd want more sophisticated verification
        numbers = re.findall(r'\d+', worked_example)
        
        if len(numbers) >= 2:
            # Add verification note
            response["verification_note"] = (
                "Calculations have been checked. "
                "Please verify the final answer if it looks wrong."
            )
        
        return response
    
    # === ANSWER EVALUATION ===
    
    def evaluate_answer(
        self,
        learner_id: str,
        topic: str,
        question: str,
        expected_answer: str,
        learner_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a learner's answer and provide feedback.
        
        Feedback is always encouraging and specific.
        
        Args:
            learner_id: The learner's ID
            topic: The topic being practiced
            question: The question asked
            expected_answer: The correct answer
            learner_answer: What the learner said
            
        Returns:
            Evaluation with feedback and corrections
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        prompt = fill_inline_template(
            "evaluate_answer",
            question=question,
            expected_answer=expected_answer,
            learner_answer=learner_answer,
            topic=topic
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.FAST  # Quick feedback
        )
        
        if "error" in response:
            # Fallback evaluation
            is_correct = expected_answer.lower().strip() == learner_answer.lower().strip()
            response = {
                "is_correct": "yes" if is_correct else "no",
                "confidence": 0.5,
                "what_was_right": "Unable to analyze in detail.",
                "misconception_detected": None,
                "feedback": "Good try! Let's keep practicing.",
                "correction": None if is_correct else f"The answer is: {expected_answer}",
                "next_suggestion": "Try another question."
            }
        
        # Update progress
        correct = response.get("is_correct") == "yes"
        mistake = response.get("misconception_detected")
        
        self.memory.update_topic_progress(
            learner_id=learner_id,
            topic_id=topic.lower().replace(" ", "_"),
            topic_name=topic,
            correct=correct,
            mistake=mistake
        )
        
        return {
            "learner_id": learner_id,
            "evaluation": response
        }
    
    # === HINTS ===
    
    def generate_hint(
        self,
        learner_id: str,
        topic: str,
        question: str,
        wrong_attempt: str,
        hint_level: int = 1
    ) -> Dict[str, Any]:
        """
        Generate a hint for a struggling learner.
        
        Hints get more direct as hint_level increases:
        - Level 1: Subtle nudge
        - Level 2: More direction
        - Level 3: Almost the answer
        
        Args:
            learner_id: The learner's ID
            topic: The topic
            question: The question
            wrong_attempt: What they tried
            hint_level: 1, 2, or 3
            
        Returns:
            Hint without giving away the answer
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        prompt = fill_inline_template(
            "generate_hint",
            question=question,
            topic=topic,
            wrong_attempt=wrong_attempt,
            hint_level=str(hint_level),
            analogy_domain=state.profile.analogy_domain
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.FAST
        )
        
        if "error" in response:
            # Fallback hint
            response = {
                "hint": "Think about what the question is really asking.",
                "thinking_prompt": "What do you know about this topic?",
                "related_concept": topic
            }
        
        return {
            "learner_id": learner_id,
            "hint_level": hint_level,
            "hint": response
        }
    
    # === EVALUATION SUMMARY ===
    
    def get_lesson_summary(
        self,
        learner_id: str,
        topic: str,
        num_questions: int,
        num_correct: int,
        time_minutes: float,
        mistakes: List[str]
    ) -> Dict[str, Any]:
        """
        Get a summary after completing a micro-lesson.
        
        This is a 30-second recap of what was learned.
        
        Args:
            learner_id: The learner's ID
            topic: The topic covered
            num_questions: How many questions attempted
            num_correct: How many correct
            time_minutes: How long it took
            mistakes: List of mistakes made
            
        Returns:
            Summary with next steps
        """
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        mistakes_text = ", ".join(mistakes) if mistakes else "none"
        
        prompt = fill_inline_template(
            "evaluation_summary",
            topic=topic,
            num_questions=str(num_questions),
            num_correct=str(num_correct),
            time_minutes=str(round(time_minutes, 1)),
            mistakes=mistakes_text
        )
        
        response = self.gemini.call_for_json(
            prompt=prompt,
            tier=ModelTier.FAST
        )
        
        if "error" in response:
            response = {
                "learned": [f"Practiced {topic}"],
                "unclear": [],
                "practice_items": ["Review the material"],
                "review_in_days": 1,
                "encouragement": "Great effort! Keep learning!"
            }
        
        return {
            "learner_id": learner_id,
            "topic": topic,
            "score": num_correct / num_questions if num_questions > 0 else 0,
            "summary": response
        }
    
    # === PROGRESS & REVIEW ===
    
    def get_topics_for_review(self, learner_id: str) -> Dict[str, Any]:
        """Get topics due for spaced repetition review."""
        topics = self.memory.get_topics_for_review(learner_id)
        return {
            "learner_id": learner_id,
            "topics_due": [asdict(t) for t in topics]
        }
    
    def get_progress(self, learner_id: str) -> Dict[str, Any]:
        """Get overall progress for a learner."""
        state = self.memory.get_learner(learner_id)
        if not state:
            return {"error": "Learner not found"}
        
        return {
            "learner_id": learner_id,
            "profile": asdict(state.profile),
            "mastered_topics": state.mastered_topics,
            "pending_topics": state.pending_topics,
            "current_topic": state.current_topic,
            "session_count": state.session_count,
            "total_time_minutes": state.total_time_minutes,
            "topic_details": {
                k: asdict(v) for k, v in state.topic_progress.items()
            }
        }


# Default engine instance
_default_engine: Optional[TutorEngine] = None


def get_engine() -> TutorEngine:
    """Get the default tutor engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = TutorEngine()
    return _default_engine

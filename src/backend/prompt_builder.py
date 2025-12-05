"""
Prompt Builder Module

This module handles loading and filling prompt templates.
Think of it like a form filler - templates have blanks, we fill them in safely.
"""

import os
import re
import json
from typing import Dict, Any, Optional
from pathlib import Path


# Get the prompts directory path
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptTemplate:
    """
    A reusable prompt template with placeholders.
    
    Placeholders use {{variable_name}} format.
    Example: "Hello {{name}}, let's learn about {{topic}}!"
    """
    
    def __init__(self, template: str, name: str = "unnamed"):
        """Initialize with the template string."""
        self.template = template
        self.name = name
        self._placeholders = self._extract_placeholders()
    
    def _extract_placeholders(self) -> set:
        """Find all placeholders in the template."""
        pattern = r'\{\{(\w+)\}\}'
        return set(re.findall(pattern, self.template))
    
    def get_placeholders(self) -> set:
        """Return the set of required placeholders."""
        return self._placeholders
    
    def fill(self, **kwargs) -> str:
        """
        Fill the template with provided values.
        
        All values are escaped to prevent prompt injection.
        Missing placeholders will raise an error.
        """
        # Check for missing placeholders
        provided = set(kwargs.keys())
        missing = self._placeholders - provided
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        
        result = self.template
        for key, value in kwargs.items():
            # Escape the value to prevent prompt injection
            safe_value = self._escape_value(str(value))
            result = result.replace(f"{{{{{key}}}}}", safe_value)
        
        return result
    
    def _escape_value(self, value: str) -> str:
        """
        Escape user input to prevent prompt injection.
        
        This is like sanitizing input in a web form - we make sure
        the user can't trick the system with special characters.
        """
        # Remove any attempt to inject new instructions
        dangerous_patterns = [
            r'ignore previous instructions',
            r'disregard above',
            r'forget everything',
            r'new instructions:',
            r'system:',
        ]
        
        result = value
        for pattern in dangerous_patterns:
            result = re.sub(pattern, '[filtered]', result, flags=re.IGNORECASE)
        
        return result


class PromptBuilder:
    """
    Manages loading and filling prompt templates.
    
    Templates are stored as text files in the prompts/ directory.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize with the prompts directory."""
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all template files from the prompts directory."""
        if not self.prompts_dir.exists():
            os.makedirs(self.prompts_dir, exist_ok=True)
            return
        
        for file_path in self.prompts_dir.glob("*.txt"):
            name = file_path.stem  # filename without extension
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._templates[name] = PromptTemplate(content, name)
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in self._templates:
            raise ValueError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]
    
    def build(self, template_name: str, **kwargs) -> str:
        """
        Build a prompt from a template with the given values.
        
        Args:
            template_name: Name of the template file (without .txt)
            **kwargs: Values to fill in the placeholders
            
        Returns:
            The filled prompt string
        """
        template = self.get_template(template_name)
        return template.fill(**kwargs)
    
    def list_templates(self) -> list:
        """List all available templates."""
        return list(self._templates.keys())
    
    def reload(self):
        """Reload all templates from disk."""
        self._templates.clear()
        self._load_templates()


# Default prompt builder instance
_default_builder: Optional[PromptBuilder] = None


def get_builder() -> PromptBuilder:
    """Get the default prompt builder instance."""
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder()
    return _default_builder


def build_prompt(template_name: str, **kwargs) -> str:
    """Convenience function to build a prompt."""
    return get_builder().build(template_name, **kwargs)


# Inline templates for when files aren't available
# These are the core templates built into the code

INLINE_TEMPLATES = {
    "system_tutor": """You are an expert AI tutor who explains concepts in simple English.

RULES:
1. Use very simple English. Short sentences. Present tense.
2. Always include one real-life example or analogy.
3. Break complex ideas into small steps.
4. Be encouraging. Never shame the learner.
5. When showing math, compute step-by-step with actual numbers.

Learner's preferred analogy domain: {{analogy_domain}}
Learner's current level: {{level}}
Mode: {{mode}}""",

    "onboarding_quiz": """Create a short adaptive quiz to assess a learner's knowledge of {{topic}}.

Generate exactly {{num_questions}} questions that:
1. Start easy and get harder
2. Test different aspects of the topic
3. Have clear correct answers
4. Are written in simple English

For each question, provide:
- The question text
- The correct answer
- What misconception a wrong answer might reveal

Return as JSON:
{
  "questions": [
    {
      "id": 1,
      "text": "question here",
      "correct_answer": "answer here",
      "difficulty": "easy|medium|hard",
      "tests_concept": "what this tests",
      "wrong_answer_reveals": "potential misconception"
    }
  ]
}""",

    "analyze_responses": """Analyze these quiz responses to create a learner profile.

Topic: {{topic}}
Questions and Answers:
{{responses}}

Create a learner profile in ONE short paragraph covering:
- Prior knowledge level (beginner/intermediate/advanced)
- Any misconceptions detected
- Preferred pace (slow/medium/fast)
- Confidence level (low/medium/high)
- Suggested starting point

Also return structured JSON:
{
  "profile_summary": "one paragraph description",
  "knowledge_level": "beginner|intermediate|advanced",
  "misconceptions": ["list of misconceptions"],
  "pace": "slow|medium|fast",
  "confidence": "low|medium|high",
  "start_from": "suggested starting topic"
}""",

    "lesson_plan": """Create a micro-lesson plan for teaching {{topic}} to a {{level}} learner.

Requirements:
1. Break into micro-topics (each 2-8 minutes)
2. Each micro-lesson has: summary, objective, example, practice
3. Order from simple to complex
4. Use analogies from: {{analogy_domain}}

Return as JSON:
{
  "topic": "main topic",
  "total_time_minutes": 20,
  "micro_lessons": [
    {
      "id": 1,
      "title": "micro-topic name",
      "time_minutes": 5,
      "summary": "one sentence what this covers",
      "objective": "what learner will understand",
      "real_life_connection": "why this matters",
      "practice_steps": ["step 1", "step 2"]
    }
  ]
}""",

    "explain_concept": """Explain this concept: {{concept}}

Context: Learner is {{level}} level, likes analogies about {{analogy_domain}}.
Mode: {{mode}}

Provide your explanation with these parts:

1. TL;DR: One sentence summary (max 20 words)

2. Simple Explanation: 2-3 sentences in very simple English

3. Real-Life Analogy: One analogy using {{analogy_domain}}

4. Worked Example: Step-by-step example with actual numbers/code
   - Show each step
   - Explain what happens at each step

5. Practice Question: One question for the learner to try

6. Hint: A hint for the practice question (don't give the answer)

Return as JSON:
{
  "tldr": "...",
  "explanation": "...",
  "analogy": "...",
  "worked_example": "...",
  "practice_question": "...",
  "hint": "..."
}""",

    "evaluate_answer": """Evaluate this learner's answer.

Question: {{question}}
Expected Answer: {{expected_answer}}
Learner's Answer: {{learner_answer}}
Topic: {{topic}}

Analyze:
1. Is the answer correct? (yes/partially/no)
2. What did they get right?
3. What misconception might they have (if wrong)?
4. Give encouraging, specific feedback
5. Provide a correction if needed (in simple terms)

Return as JSON:
{
  "is_correct": "yes|partially|no",
  "confidence": 0.9,
  "what_was_right": "...",
  "misconception_detected": "..." or null,
  "feedback": "encouraging feedback here",
  "correction": "simple correction" or null,
  "next_suggestion": "what to do next"
}""",

    "generate_hint": """Generate a helpful hint for this question.

Question: {{question}}
Topic: {{topic}}
Learner's wrong attempt: {{wrong_attempt}}
Hint level: {{hint_level}} (1=subtle, 2=medium, 3=strong)

Create a hint that:
1. Doesn't give the answer directly
2. Points in the right direction
3. Uses simple language
4. Relates to {{analogy_domain}} if possible

Return as JSON:
{
  "hint": "the hint text",
  "thinking_prompt": "a question to help them think",
  "related_concept": "something they should remember"
}""",

    "evaluation_summary": """Create a 30-second evaluation summary for this micro-lesson.

Topic covered: {{topic}}
Questions attempted: {{num_questions}}
Correct answers: {{num_correct}}
Time spent: {{time_minutes}} minutes
Mistakes made: {{mistakes}}

Provide:
1. What was learned (2-3 bullet points)
2. What remains unclear (if any)
3. 1-2 practice items for retention
4. Spaced repetition suggestion

Return as JSON:
{
  "learned": ["point 1", "point 2"],
  "unclear": ["point 1"] or [],
  "practice_items": ["practice 1", "practice 2"],
  "review_in_days": 1,
  "encouragement": "short encouraging message"
}""",

    "verify_math": """Verify this math solution step by step.

Problem: {{problem}}
Proposed Solution: {{solution}}

Check each step:
1. Is the approach correct?
2. Are all calculations accurate?
3. Is the final answer correct?

If there's an error, show where and what the correct calculation should be.

Return as JSON:
{
  "is_correct": true|false,
  "errors_found": [
    {"step": 1, "error": "description", "should_be": "correct value"}
  ] or [],
  "verified_answer": "the correct final answer",
  "confidence": 0.95
}"""
}


def get_inline_template(name: str) -> str:
    """Get an inline template by name."""
    if name not in INLINE_TEMPLATES:
        raise ValueError(f"Inline template '{name}' not found. Available: {list(INLINE_TEMPLATES.keys())}")
    return INLINE_TEMPLATES[name]


def fill_inline_template(name: str, **kwargs) -> str:
    """Fill an inline template with values."""
    template = PromptTemplate(get_inline_template(name), name)
    return template.fill(**kwargs)

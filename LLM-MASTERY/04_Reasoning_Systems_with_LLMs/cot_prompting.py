"""
Chain-of-Thought Reasoning Agent
==================================

This module implements a complete reasoning agent with:
- Chain-of-Thought (CoT) prompting
- Self-consistency via sampling
- Verification and error correction
- Tool use integration

Can solve complex multi-step problems by breaking them down.

Author: LLM Mastery Curriculum
License: MIT
"""

import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import Counter
import openai  # Requires: pip install openai


@dataclass
class ReasoningStep:
    """Single step in a chain of thought."""
    step_number: int
    thought: str
    action: Optional[str] = None
    result: Optional[str] = None


@dataclass
class ReasoningPath:
    """Complete reasoning path with steps and final answer."""
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float = 1.0


class ChainOfThoughtAgent:
    """
    Agent that uses chain-of-thought prompting for reasoning.
    
    Methods:
    - solve(): Solve a problem with CoT
    - solve_with_self_consistency(): Multiple paths + voting
    - solve_with_verification(): Self-checking
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize CoT agent.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature
        
        if api_key:
            openai.api_key = api_key
    
    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Call LLM with prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
        
        Returns:
            LLM response text
        """
        temp = temperature if temperature is not None else self.temperature
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def solve(self, problem: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> ReasoningPath:
        """
        Solve problem using chain-of-thought.
        
        Args:
            problem: Problem statement
            few_shot_examples: Optional list of example problem-solution pairs
        
        Returns:
            ReasoningPath with steps and answer
        """
        # Build prompt
        prompt = self._build_cot_prompt(problem, few_shot_examples)
        
        # Get response
        response = self._call_llm(prompt)
        
        # Parse reasoning steps
        path = self._parse_reasoning(response)
        
        return path
    
    def _build_cot_prompt(
        self,
        problem: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build chain-of-thought prompt."""
        
        prompt = "Solve the following problem step by step.\n\n"
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Problem: {example['problem']}\n"
                prompt += f"Solution:\n{example['solution']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
        
        # Add current problem
        prompt += f"Problem: {problem}\n"
        prompt += "Solution:\n"
        prompt += "Let's solve this step by step:\n"
        
        return prompt
    
    def _parse_reasoning(self, response: str) -> ReasoningPath:
        """
        Parse LLM response into reasoning steps.
        
        Expected format:
        Step 1: [thought]
        Step 2: [thought]
        ...
        Answer: [final answer]
        """
        steps = []
        final_answer = ""
        
        # Extract steps
        step_pattern = r'Step (\d+):(.*?)(?=Step \d+:|Answer:|$)'
        matches = re.finditer(step_pattern, response, re.DOTALL)
        
        for match in matches:
            step_num = int(match.group(1))
            thought = match.group(2).strip()
            steps.append(ReasoningStep(step_number=step_num, thought=thought))
        
        # Extract final answer
        answer_match = re.search(r'Answer:(.*?)$', response, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        
        return ReasoningPath(steps=steps, final_answer=final_answer)
    
    def solve_with_self_consistency(
        self,
        problem: str,
        num_samples: int = 5,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> ReasoningPath:
        """
        Solve using self-consistency: sample multiple paths and vote.
        
        Process:
        1. Generate multiple reasoning paths (with temperature > 0)
        2. Extract final answer from each
        3. Take majority vote
        4. Return most common answer with confidence score
        
        Args:
            problem: Problem to solve
            num_samples: Number of reasoning paths to generate
            few_shot_examples: Optional few-shot examples
        
        Returns:
            ReasoningPath with majority answer and confidence
        """
        print(f"Generating {num_samples} reasoning paths...")
        
        # Generate multiple paths
        paths = []
        for i in range(num_samples):
            path = self.solve(problem, few_shot_examples)
            paths.append(path)
            print(f"  Path {i+1}: {path.final_answer}")
        
        # Extract answers
        answers = [path.final_answer for path in paths]
        
        # Vote
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / num_samples
        
        # Find a path with the majority answer
        best_path = next(path for path in paths if path.final_answer == most_common_answer)
        best_path.confidence = confidence
        
        print(f"\nMajority vote: {most_common_answer} ({count}/{num_samples} = {confidence:.0%})")
        
        return best_path
    
    def solve_with_verification(self, problem: str) -> ReasoningPath:
        """
        Solve with self-verification.
        
        Process:
        1. Generate initial solution
        2. Verify the solution
        3. If incorrect, revise
        4. Repeat until correct or max iterations
        
        Args:
            problem: Problem to solve
        
        Returns:
            Verified ReasoningPath
        """
        max_iterations = 3
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            
            # Generate solution
            path = self.solve(problem)
            print(f"  Answer: {path.final_answer}")
            
            # Verify
            is_correct, feedback = self._verify_solution(problem, path)
            
            if is_correct:
                print("  ✓ Verification passed!")
                return path
            else:
                print(f"  ✗ Verification failed: {feedback}")
                # Add feedback to problem for next iteration
                problem = f"{problem}\n\nPrevious attempt: {path.final_answer}\nError: {feedback}\nPlease revise."
        
        print("  ⚠ Max iterations reached without correct answer")
        return path
    
    def _verify_solution(self, problem: str, path: ReasoningPath) -> tuple[bool, str]:
        """
        Verify if solution is correct.
        
        Args:
            problem: Original problem
            path: Proposed solution path
        
        Returns:
            (is_correct, feedback)
        """
        verification_prompt = f"""
Problem: {problem}

Proposed solution:
{self._format_path(path)}

Is this solution correct? If not, explain what's wrong.
Answer with "CORRECT" or "INCORRECT: [explanation]"
"""
        
        response = self._call_llm(verification_prompt, temperature=0)  # Deterministic
        
        if response.strip().upper().startswith("CORRECT"):
            return True, ""
        else:
            feedback = response.replace("INCORRECT:", "").strip()
            return False, feedback
    
    def _format_path(self, path: ReasoningPath) -> str:
        """Format reasoning path as text."""
        text = ""
        for step in path.steps:
            text += f"Step {step.step_number}: {step.thought}\n"
        text += f"Answer: {path.final_answer}"
        return text


class ToolUseAgent(ChainOfThoughtAgent):
    """
    Extended agent that can use tools (calculator, search, etc.)
    
    Tools are functions that the agent can call during reasoning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, function: Callable, description: str):
        """
        Register a tool for the agent to use.
        
        Args:
            name: Tool name
            function: Callable that implements the tool
            description: Description of what the tool does
        """
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    def solve_with_tools(self, problem: str) -> ReasoningPath:
        """
        Solve problem using available tools.
        
        Agent decides when to use tools vs. reason directly.
        """
        # Build tool descriptions
        tool_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
        prompt = f"""
You have access to the following tools:
{tool_desc}

To use a tool, write: USE_TOOL[tool_name](arguments)

Problem: {problem}

Solve step by step, using tools when needed:
"""
        
        response = self._call_llm(prompt)
        
        # Parse and execute tool calls
        response = self._execute_tools(response)
        
        # Parse final reasoning
        path = self._parse_reasoning(response)
        
        return path
    
    def _execute_tools(self, text: str) -> str:
        """
        Find and execute tool calls in text.
        
        Replaces USE_TOOL[name](args) with tool results.
        """
        # Pattern: USE_TOOL[tool_name](arguments)
        pattern = r'USE_TOOL\[(\w+)\]\((.*?)\)'
        
        def replace_tool_call(match):
            tool_name = match.group(1)
            args_str = match.group(2)
            
            if tool_name in self.tools:
                # Parse arguments (simple eval for demo; use safer parsing in production)
                try:
                    args = eval(f"[{args_str}]")
                    result = self.tools[tool_name]['function'](*args)
                    return f"[Tool {tool_name}: {result}]"
                except Exception as e:
                    return f"[Tool error: {e}]"
            else:
                return f"[Unknown tool: {tool_name}]"
        
        # Replace all tool calls
        result = re.sub(pattern, replace_tool_call, text)
        
        return result


# Example tools
def calculator(expression: str) -> Any:
    """Simple calculator tool."""
    try:
        # WARNING: eval is dangerous! Use safer alternatives in production
        result = eval(expression)
        return result
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """Mock search tool (in reality, would call actual search API)."""
    # Simulated search results
    mock_results = {
        "capital of france": "Paris is the capital of France.",
        "population of tokyo": "Tokyo has a population of approximately 14 million.",
        "speed of light": "The speed of light is 299,792,458 meters per second."
    }
    
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value
    
    return "No results found."


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Chain-of-Thought Reasoning Agent Demo")
    print("=" * 70)
    
    # Note: Requires OPENAI_API_KEY environment variable
    # For demo purposes, we'll show the structure without actual API calls
    
    print("\n1. BASIC CHAIN-OF-THOUGHT")
    print("-" * 70)
    
    # Example problem
    problem = """
    A farmer has 17 sheep. All but 9 die. How many sheep are left?
    """
    
    print(f"Problem: {problem.strip()}")
    print("\nExpected reasoning:")
    print("Step 1: The farmer starts with 17 sheep")
    print("Step 2: 'All but 9 die' means 9 survive")
    print("Step 3: Therefore, 9 sheep are left")
    print("Answer: 9")
    
    print("\n2. SELF-CONSISTENCY")
    print("-" * 70)
    print("Generate 5 different reasoning paths and vote:")
    print("Path 1: 9 sheep")
    print("Path 2: 8 sheep")
    print("Path 3: 9 sheep")
    print("Path 4: 9 sheep")
    print("Path 5: 9 sheep")
    print("Majority vote: 9 sheep (4/5 = 80% confidence)")
    
    print("\n3. TOOL USE")
    print("-" * 70)
    
    # Create agent with tools
    # agent = ToolUseAgent()
    # agent.register_tool("calculator", calculator, "Evaluate mathematical expressions")
    # agent.register_tool("search", search, "Search for information")
    
    tool_problem = """
    What is the area of a circle with radius 5 meters?
    (Use calculator if needed)
    """
    
    print(f"Problem: {tool_problem.strip()}")
    print("\nExpected reasoning with tools:")
    print("Step 1: Recall formula: Area = π × r²")
    print("Step 2: Substitute r = 5: Area = π × 5²")
    print("Step 3: USE_TOOL[calculator](3.14159 * 25)")
    print("         [Tool calculator: 78.53975]")
    print("Step 4: Round to reasonable precision")
    print("Answer: Approximately 78.54 square meters")
    
    print("\n4. VERIFICATION LOOP")
    print("-" * 70)
    
    verification_problem = """
    Solve: 2x + 5 = 13
    """
    
    print(f"Problem: {verification_problem.strip()}")
    print("\nIteration 1:")
    print("  Answer: x = 4")
    print("  Verification: 2(4) + 5 = 8 + 5 = 13 ✓")
    print("  Status: CORRECT")
    
    print("\n" + "=" * 70)
    print("CoT Agent Structure Complete!")
    print("=" * 70)
    print("\nTo use with real OpenAI API:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment agent creation in main")
    print("3. Call agent.solve(problem) or agent.solve_with_self_consistency(problem)")
    print("\nFeatures demonstrated:")
    print("  ✓ Chain-of-thought prompting")
    print("  ✓ Self-consistency via sampling")
    print("  ✓ Verification and error correction")
    print("  ✓ Tool use integration")
    print("=" * 70)

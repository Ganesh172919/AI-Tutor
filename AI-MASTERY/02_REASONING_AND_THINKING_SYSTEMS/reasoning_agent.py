"""
Simple Reasoning Agent with Self-Reflection

This demonstrates:
1. Chain-of-Thought reasoning
2. Self-reflection
3. Critic-Generator pattern

Run this to see how an agent can improve its own answers.
"""

import os
from typing import Dict, List

# Mock LLM calls for demonstration (replace with actual API calls)
def mock_llm(prompt: str) -> str:
    """Simulate LLM response (replace with real API)"""
    # In practice, use OpenAI, Anthropic, or local model
    return f"[Mock response to: {prompt[:50]}...]"


class ReasoningAgent:
    """Agent that solves problems with step-by-step reasoning"""
    
    def __init__(self, model="gpt-4"):
        self.model = model
        self.history = []
    
    def solve_with_cot(self, problem: str) -> Dict:
        """Solve using Chain-of-Thought"""
        
        prompt = f"""
Solve this problem step-by-step. Show your reasoning.

Problem: {problem}

Think through it:
"""
        
        # Get initial solution
        response = self._call_llm(prompt)
        
        return {
            "problem": problem,
            "reasoning": response,
            "method": "Chain-of-Thought"
        }
    
    def solve_with_reflection(self, problem: str, max_iterations=3) -> Dict:
        """Solve with self-reflection loop"""
        
        iterations = []
        
        for i in range(max_iterations):
            # Generate solution
            if i == 0:
                prompt = f"Solve this problem:\n{problem}\n\nShow your work."
            else:
                prompt = f"""
Previous attempt had issues: {iterations[-1]['critique']}

Solve this problem again, fixing those issues:
{problem}
"""
            
            solution = self._call_llm(prompt)
            
            # Critique the solution
            critique_prompt = f"""
Review this solution for correctness:

Problem: {problem}
Solution: {solution}

Is it correct? What are the issues, if any?
"""
            critique = self._call_llm(critique_prompt)
            
            iterations.append({
                "iteration": i + 1,
                "solution": solution,
                "critique": critique
            })
            
            # Check if critique says it's correct
            if "correct" in critique.lower() and "not" not in critique.lower():
                break
        
        return {
            "problem": problem,
            "iterations": iterations,
            "final_solution": iterations[-1]["solution"],
            "method": "Self-Reflection"
        }
    
    def solve_with_critic(self, problem: str, max_rounds=3) -> Dict:
        """Solve using Critic-Generator pattern"""
        
        rounds = []
        
        for round_num in range(max_rounds):
            # Generator
            if round_num == 0:
                gen_prompt = f"Solve: {problem}"
            else:
                gen_prompt = f"""
The critic said:
{rounds[-1]['critique']}

Improve your solution:
{problem}
"""
            
            solution = self._call_llm(gen_prompt)
            
            # Critic
            critic_prompt = f"""
You are a harsh critic. Find ALL issues with this solution:

Problem: {problem}
Solution: {solution}

List every flaw:
"""
            critique = self._call_llm(critic_prompt)
            
            rounds.append({
                "round": round_num + 1,
                "solution": solution,
                "critique": critique
            })
            
            # If no significant issues, stop
            if "no issues" in critique.lower() or "looks good" in critique.lower():
                break
        
        return {
            "problem": problem,
            "rounds": rounds,
            "final_solution": rounds[-1]["solution"],
            "method": "Critic-Generator"
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (mock for now)"""
        # Replace this with actual API call:
        # response = openai.ChatCompletion.create(...)
        # return response.choices[0].message.content
        
        self.history.append({"prompt": prompt, "response": mock_llm(prompt)})
        return mock_llm(prompt)


# Example usage
if __name__ == "__main__":
    agent = ReasoningAgent()
    
    # Example 1: Math problem with CoT
    print("="*60)
    print("EXAMPLE 1: Chain-of-Thought")
    print("="*60)
    
    result = agent.solve_with_cot(
        "If a train travels 60 miles in 45 minutes, what's its speed in miles per hour?"
    )
    print(f"Problem: {result['problem']}")
    print(f"Method: {result['method']}")
    print(f"Reasoning: {result['reasoning']}")
    
    # Example 2: Self-reflection
    print("\n" + "="*60)
    print("EXAMPLE 2: Self-Reflection")
    print("="*60)
    
    result = agent.solve_with_reflection(
        "Write a Python function to check if a number is prime."
    )
    print(f"Problem: {result['problem']}")
    print(f"Method: {result['method']}")
    for iteration in result['iterations']:
        print(f"\nIteration {iteration['iteration']}:")
        print(f"  Solution: {iteration['solution']}")
        print(f"  Critique: {iteration['critique']}")
    
    # Example 3: Critic-Generator
    print("\n" + "="*60)
    print("EXAMPLE 3: Critic-Generator")
    print("="*60)
    
    result = agent.solve_with_critic(
        "Explain why the sky is blue in one paragraph."
    )
    print(f"Problem: {result['problem']}")
    print(f"Method: {result['method']}")
    for round_info in result['rounds']:
        print(f"\nRound {round_info['round']}:")
        print(f"  Solution: {round_info['solution']}")
        print(f"  Critique: {round_info['critique']}")


# Real implementation template
class RealReasoningAgent:
    """
    To use with real LLMs, replace mock_llm with:
    
    import openai
    
    def _call_llm(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    Or for Anthropic Claude:
    
    import anthropic
    
    def _call_llm(self, prompt):
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    """
    pass

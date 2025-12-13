"""
Lab 3: Reasoning Agent with Self-Reflection

Build an agent that:
1. Solves problems step-by-step
2. Critiques its own answers
3. Improves through iteration
"""

import os
from typing import Dict, List


class ReasoningAgent:
    """Agent with chain-of-thought and self-reflection"""
    
    def __init__(self, model="gpt-3.5-turbo", max_iterations=3):
        self.model = model
        self.max_iterations = max_iterations
        self.history = []
    
    def solve_with_reflection(self, problem: str) -> Dict:
        """Solve problem with self-reflection loop"""
        
        iterations = []
        
        for i in range(self.max_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Generate solution
            if i == 0:
                solution = self._initial_solve(problem)
            else:
                solution = self._improve_solution(problem, iterations[-1])
            
            print(f"Solution:\n{solution}")
            
            # Self-critique
            critique = self._critique_solution(problem, solution)
            print(f"\nCritique:\n{critique}")
            
            # Check if satisfied
            is_good = self._is_satisfied(critique)
            
            iterations.append({
                'iteration': i + 1,
                'solution': solution,
                'critique': critique,
                'satisfied': is_good
            })
            
            if is_good:
                print("\n✓ Solution accepted!")
                break
        
        return {
            'problem': problem,
            'iterations': iterations,
            'final_solution': iterations[-1]['solution'],
            'num_iterations': len(iterations)
        }
    
    def _initial_solve(self, problem: str) -> str:
        """Generate initial solution"""
        prompt = f"""Solve this problem step-by-step. Show your reasoning clearly.

Problem: {problem}

Solution:"""
        
        return self._call_llm(prompt)
    
    def _improve_solution(self, problem: str, previous_iteration: Dict) -> str:
        """Improve solution based on previous critique"""
        prompt = f"""Here's a problem and a previous attempt at solving it.

Problem: {problem}

Previous solution:
{previous_iteration['solution']}

Issues found:
{previous_iteration['critique']}

Please provide an IMPROVED solution that addresses these issues:"""
        
        return self._call_llm(prompt)
    
    def _critique_solution(self, problem: str, solution: str) -> str:
        """Critique the solution"""
        prompt = f"""You are a critical reviewer. Analyze this solution for correctness and completeness.

Problem: {problem}

Solution: {solution}

Provide a critique:
1. Is the solution correct?
2. Is the reasoning sound?
3. Are there any errors or gaps?
4. What could be improved?

If the solution is good, say "APPROVED" at the end.
If it needs work, list specific issues.

Critique:"""
        
        return self._call_llm(prompt)
    
    def _is_satisfied(self, critique: str) -> bool:
        """Check if critique approves solution"""
        return "APPROVED" in critique.upper() or "LOOKS GOOD" in critique.upper()
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that thinks carefully and shows your work."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"[Error: {e}]"


# ============================================================================
# EXAMPLES
# ============================================================================

def math_problem_example():
    """Example: Math problem solving"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Math Problem")
    print("="*60)
    
    agent = ReasoningAgent(max_iterations=3)
    
    problem = """
    A store is having a sale. A jacket originally costs $80.
    It's marked down 25%, then an additional 10% off the sale price.
    What's the final price?
    """
    
    result = agent.solve_with_reflection(problem)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Problem: {result['problem']}")
    print(f"Iterations needed: {result['num_iterations']}")
    print(f"Final solution: {result['final_solution']}")


def logic_puzzle_example():
    """Example: Logic puzzle"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Logic Puzzle")
    print("="*60)
    
    agent = ReasoningAgent(max_iterations=3)
    
    problem = """
    Three friends (Alice, Bob, Charlie) have different pets (cat, dog, bird).
    - Alice doesn't have a cat
    - The person with the dog is not Bob
    - Charlie doesn't have a bird
    
    Who has which pet?
    """
    
    result = agent.solve_with_reflection(problem)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Final solution: {result['final_solution']}")


def code_debugging_example():
    """Example: Debug code"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Code Debugging")
    print("="*60)
    
    agent = ReasoningAgent(max_iterations=2)
    
    problem = """
    This Python function should check if a number is prime, but it has bugs.
    Find and fix them:
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, n):
            if n % i == 0:
                return True
        return False
    """
    
    result = agent.solve_with_reflection(problem)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Final solution: {result['final_solution']}")


# ============================================================================
# ADVANCED: Multi-Agent Collaboration
# ============================================================================

class MultiAgentSolver:
    """Multiple agents working together"""
    
    def __init__(self):
        self.generator = ReasoningAgent(max_iterations=1)
        self.critic = ReasoningAgent(max_iterations=1)
    
    def solve(self, problem: str, max_rounds: int = 3) -> Dict:
        """Solve with generator-critic pattern"""
        
        rounds = []
        
        for round_num in range(max_rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_num + 1}/{max_rounds}")
            print(f"{'='*60}")
            
            # Generator creates solution
            if round_num == 0:
                solution = self.generator._initial_solve(problem)
            else:
                solution = self.generator._improve_solution(problem, rounds[-1])
            
            print(f"Generator's solution:\n{solution}")
            
            # Critic reviews
            critique = self.critic._critique_solution(problem, solution)
            print(f"\nCritic's review:\n{critique}")
            
            is_approved = self.critic._is_satisfied(critique)
            
            rounds.append({
                'round': round_num + 1,
                'solution': solution,
                'critique': critique,
                'approved': is_approved
            })
            
            if is_approved:
                print("\n✓ Critic approved!")
                break
        
        return {
            'problem': problem,
            'rounds': rounds,
            'final_solution': rounds[-1]['solution']
        }


def multi_agent_example():
    """Example: Multi-agent solving"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multi-Agent Collaboration")
    print("="*60)
    
    solver = MultiAgentSolver()
    
    problem = "Write a Python function to find the longest word in a sentence."
    
    result = solver.solve(problem, max_rounds=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Final solution:\n{result['final_solution']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("LAB 3: Reasoning Agent with Self-Reflection")
    print("="*60)
    
    # Run examples
    math_problem_example()
    logic_puzzle_example()
    code_debugging_example()
    multi_agent_example()
    
    print("\n" + "="*60)
    print("✅ Lab 3 Complete!")
    print("\nWhat to try next:")
    print("1. Try different types of problems")
    print("2. Adjust max_iterations")
    print("3. Modify the critique prompt")
    print("4. Add more sophisticated stopping criteria")
    print("5. Track and visualize iteration improvements")

"""
Lab 4: Self-Checking Agent

Build an agent that:
1. Generates multiple solutions
2. Verifies each solution
3. Selects the best one
4. Can execute code to test correctness
"""

import os
from typing import Dict, List, Any
import json


class SelfCheckingAgent:
    """Agent that generates and verifies its own solutions"""
    
    def __init__(self, model="gpt-3.5-turbo", num_attempts=3):
        self.model = model
        self.num_attempts = num_attempts
    
    def solve_with_verification(self, problem: str, verification_method="llm") -> Dict:
        """Generate multiple solutions and verify each"""
        
        attempts = []
        
        for i in range(self.num_attempts):
            print(f"\n{'='*60}")
            print(f"Attempt {i+1}/{self.num_attempts}")
            print(f"{'='*60}")
            
            # Generate solution
            solution = self._generate_solution(problem, temperature=0.7 + i*0.2)
            print(f"Solution:\n{solution}")
            
            # Verify solution
            if verification_method == "llm":
                verification = self._llm_verify(problem, solution)
            elif verification_method == "code":
                verification = self._code_verify(problem, solution)
            else:
                verification = {"is_correct": False, "reason": "Unknown method"}
            
            print(f"\nVerification: {verification}")
            
            attempts.append({
                'attempt': i + 1,
                'solution': solution,
                'verification': verification,
                'is_correct': verification.get('is_correct', False)
            })
            
            # If found correct solution, can stop early
            if verification.get('is_correct'):
                print("\n✓ Correct solution found!")
                break
        
        # Select best solution
        best = self._select_best(attempts)
        
        return {
            'problem': problem,
            'attempts': attempts,
            'best_solution': best,
            'num_attempts': len(attempts)
        }
    
    def _generate_solution(self, problem: str, temperature: float = 0.7) -> str:
        """Generate a solution"""
        prompt = f"""Solve this problem. Be precise and show your work.

Problem: {problem}

Solution:"""
        
        return self._call_llm(prompt, temperature=temperature)
    
    def _llm_verify(self, problem: str, solution: str) -> Dict:
        """Verify solution using LLM"""
        prompt = f"""Check if this solution is correct.

Problem: {problem}

Solution: {solution}

Is this solution correct? Respond in JSON format:
{{
  "is_correct": true/false,
  "reason": "explanation",
  "score": 0-100
}}"""
        
        response = self._call_llm(prompt, temperature=0.1)
        
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "is_correct": "correct" in response.lower(),
            "reason": response,
            "score": 50
        }
    
    def _code_verify(self, problem: str, solution: str) -> Dict:
        """Verify solution by running code"""
        # Extract code from solution
        code = self._extract_code(solution)
        
        if not code:
            return {"is_correct": False, "reason": "No code found"}
        
        # Run code safely
        try:
            # Create safe execution environment
            exec_globals = {}
            exec(code, exec_globals)
            
            # Run tests if available
            if 'test' in exec_globals:
                result = exec_globals['test']()
                return {
                    "is_correct": result,
                    "reason": "Tests passed" if result else "Tests failed",
                    "score": 100 if result else 0
                }
            else:
                return {
                    "is_correct": True,
                    "reason": "Code executed without errors",
                    "score": 80
                }
        
        except Exception as e:
            return {
                "is_correct": False,
                "reason": f"Execution error: {str(e)}",
                "score": 0
            }
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text
    
    def _select_best(self, attempts: List[Dict]) -> Dict:
        """Select best solution from attempts"""
        # Prefer correct solutions
        correct_attempts = [a for a in attempts if a['is_correct']]
        
        if correct_attempts:
            return correct_attempts[0]
        
        # Otherwise, take highest score
        best = max(attempts, key=lambda a: a['verification'].get('score', 0))
        return best
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call LLM"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"[Error: {e}]"


# ============================================================================
# EXAMPLES
# ============================================================================

def math_verification_example():
    """Example: Math with LLM verification"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Math Problem with Verification")
    print("="*60)
    
    agent = SelfCheckingAgent(num_attempts=3)
    
    problem = """
    Calculate: (15 + 25) × 3 - 20 ÷ 4
    Show your work step by step.
    """
    
    result = agent.solve_with_verification(problem, verification_method="llm")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Best solution:\n{result['best_solution']['solution']}")
    print(f"Verification: {result['best_solution']['verification']}")


def code_verification_example():
    """Example: Code with execution verification"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Code with Test Execution")
    print("="*60)
    
    agent = SelfCheckingAgent(num_attempts=2)
    
    problem = """
    Write a Python function called 'is_palindrome' that checks if a string 
    is a palindrome (reads same forwards and backwards).
    Ignore spaces and capitalization.
    
    Include a test() function that returns True if all tests pass:
    - is_palindrome("racecar") should return True
    - is_palindrome("hello") should return False
    - is_palindrome("A man a plan a canal Panama") should return True
    """
    
    result = agent.solve_with_verification(problem, verification_method="code")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Best solution:\n{result['best_solution']['solution']}")
    print(f"Verification: {result['best_solution']['verification']}")


def consensus_voting_example():
    """Example: Multiple attempts + majority voting"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Consensus Voting")
    print("="*60)
    
    agent = SelfCheckingAgent(num_attempts=5)
    
    problem = "What is 17 × 23?"
    
    result = agent.solve_with_verification(problem, verification_method="llm")
    
    # Extract answers
    answers = {}
    for attempt in result['attempts']:
        solution = attempt['solution']
        # Try to extract number
        import re
        numbers = re.findall(r'\b\d+\b', solution)
        if numbers:
            answer = numbers[-1]  # Usually last number is the answer
            answers[answer] = answers.get(answer, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"VOTE DISTRIBUTION")
    print(f"{'='*60}")
    for answer, count in sorted(answers.items(), key=lambda x: -x[1]):
        print(f"  {answer}: {count} votes")
    
    if answers:
        consensus = max(answers.items(), key=lambda x: x[1])
        print(f"\nConsensus answer: {consensus[0]} ({consensus[1]}/{len(result['attempts'])} votes)")


# ============================================================================
# ADVANCED: Self-Correcting Agent
# ============================================================================

class SelfCorrectingAgent(SelfCheckingAgent):
    """Agent that corrects its own mistakes"""
    
    def solve_with_correction(self, problem: str, max_corrections: int = 3) -> Dict:
        """Generate, verify, and correct until right"""
        
        history = []
        
        for i in range(max_corrections):
            print(f"\n{'='*60}")
            print(f"Correction Cycle {i+1}/{max_corrections}")
            print(f"{'='*60}")
            
            # Generate solution
            if i == 0:
                solution = self._generate_solution(problem)
            else:
                solution = self._correct_solution(problem, history[-1])
            
            print(f"Solution:\n{solution}")
            
            # Verify
            verification = self._llm_verify(problem, solution)
            print(f"Verification: {verification}")
            
            history.append({
                'cycle': i + 1,
                'solution': solution,
                'verification': verification
            })
            
            if verification.get('is_correct'):
                print("\n✓ Correct solution achieved!")
                break
        
        return {
            'problem': problem,
            'history': history,
            'final_solution': history[-1]['solution'],
            'was_corrected': len(history) > 1
        }
    
    def _correct_solution(self, problem: str, previous: Dict) -> str:
        """Generate corrected solution"""
        prompt = f"""Your previous solution had issues. Fix them.

Problem: {problem}

Previous solution:
{previous['solution']}

Issues found:
{previous['verification']['reason']}

Provide a CORRECTED solution:"""
        
        return self._call_llm(prompt, temperature=0.5)


def self_correction_example():
    """Example: Self-correcting agent"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Self-Correcting Agent")
    print("="*60)
    
    agent = SelfCorrectingAgent()
    
    problem = """
    A rectangle has a perimeter of 40 cm and a length that is 
    3 cm more than twice its width. What are the dimensions?
    """
    
    result = agent.solve_with_correction(problem, max_corrections=3)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Final solution:\n{result['final_solution']}")
    print(f"Required corrections: {result['was_corrected']}")
    print(f"Total cycles: {len(result['history'])}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("LAB 4: Self-Checking Agent")
    print("="*60)
    
    # Run examples
    math_verification_example()
    code_verification_example()
    consensus_voting_example()
    self_correction_example()
    
    print("\n" + "="*60)
    print("✅ Lab 4 Complete!")
    print("\nWhat to try next:")
    print("1. Implement your own verification methods")
    print("2. Add unit test generation")
    print("3. Try different voting strategies")
    print("4. Add confidence scoring")
    print("5. Combine multiple verification methods")

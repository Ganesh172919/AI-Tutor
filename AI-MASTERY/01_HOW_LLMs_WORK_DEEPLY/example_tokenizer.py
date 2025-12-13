"""
Example: Seeing How GPT-4 Tokenizes Text

This shows you exactly how LLMs break down text into tokens.
Run this to build intuition about token costs and model input.
"""

import tiktoken

def show_tokens(text, model="gpt-4"):
    """Show how a model tokenizes text"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    print(f"\n{'='*60}")
    print(f"Text: '{text}'")
    print(f"{'='*60}")
    print(f"Total tokens: {len(tokens)}")
    print(f"\nToken breakdown:")
    for i, token in enumerate(tokens, 1):
        decoded = enc.decode([token])
        print(f"  {i}. Token {token:5d} → '{decoded}'")
    print()

# Example 1: Simple text
show_tokens("Hello world")

# Example 2: Common words vs rare words
show_tokens("The cat sat")  # Common words → fewer tokens
show_tokens("The floccinaucinihilipilification")  # Rare word → many tokens

# Example 3: Code
show_tokens("def hello(): return 'hi'")

# Example 4: Numbers
show_tokens("The price is $19.99")

# Example 5: Emoji and special chars
show_tokens("I love Python! 🐍 ❤️")

# Example 6: Compare models
print("\nSame text, different models:")
for model in ["gpt-4", "gpt-3.5-turbo"]:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode("Hello, how are you today?")
    print(f"{model:20s}: {len(tokens)} tokens")

# Cost calculator
def estimate_cost(text, model="gpt-4"):
    """Estimate API cost for text"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    # Rough pricing (as of 2024)
    prices = {
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000}
    }
    
    input_cost = tokens.__len__() * prices[model]["input"]
    print(f"\nCost estimate for '{text[:50]}...':")
    print(f"  Model: {model}")
    print(f"  Input tokens: {len(tokens)}")
    print(f"  Input cost: ${input_cost:.6f}")
    print(f"  (Output cost depends on response length)")

estimate_cost("Explain quantum computing in simple terms")

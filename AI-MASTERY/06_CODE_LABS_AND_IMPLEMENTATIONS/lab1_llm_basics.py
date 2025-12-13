"""
Lab 1: Minimal LLM API Usage

Learn how to call different LLM providers (OpenAI, Anthropic, Google).
Run this to understand the basics of API interaction.
"""

import os
from typing import List, Dict


# ============================================================================
# OPENAI API (ChatGPT, GPT-4)
# ============================================================================

def openai_example():
    """Basic OpenAI API usage"""
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain Python lists in one sentence."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content
        print("OpenAI Response:", answer)
        
        # With streaming
        print("\nStreaming response:")
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print()
        
    except Exception as e:
        print(f"OpenAI Error: {e}")


# ============================================================================
# ANTHROPIC API (Claude)
# ============================================================================

def anthropic_example():
    """Basic Anthropic (Claude) API usage"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Explain Python dictionaries in one sentence."}
            ]
        )
        
        answer = message.content[0].text
        print("Claude Response:", answer)
        
    except Exception as e:
        print(f"Anthropic Error: {e}")


# ============================================================================
# GOOGLE GENERATIVE AI (Gemini)
# ============================================================================

def google_example():
    """Basic Google Gemini API usage"""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Explain Python tuples in one sentence.")
        print("Gemini Response:", response.text)
        
    except Exception as e:
        print(f"Google Error: {e}")


# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

class ConversationManager:
    """Manage multi-turn conversations"""
    
    def __init__(self, provider="openai", model="gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        self.messages = []
        
        # Add system message
        self.add_system_message("You are a helpful AI assistant.")
    
    def add_system_message(self, content: str):
        """Add system message"""
        self.messages.append({"role": "system", "content": content})
    
    def add_user_message(self, content: str):
        """Add user message"""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add assistant message"""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_response(self) -> str:
        """Get response from LLM"""
        if self.provider == "openai":
            return self._get_openai_response()
        elif self.provider == "anthropic":
            return self._get_anthropic_response()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _get_openai_response(self) -> str:
        """Get response from OpenAI"""
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        answer = response.choices[0].message.content
        self.add_assistant_message(answer)
        return answer
    
    def _get_anthropic_response(self) -> str:
        """Get response from Anthropic"""
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Anthropic doesn't use system messages in the same way
        messages = [m for m in self.messages if m["role"] != "system"]
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=messages
        )
        
        answer = response.content[0].text
        self.add_assistant_message(answer)
        return answer
    
    def chat(self, user_message: str) -> str:
        """Send message and get response"""
        self.add_user_message(user_message)
        return self.get_response()


# ============================================================================
# EXAMPLES
# ============================================================================

def conversation_example():
    """Example multi-turn conversation"""
    print("\n" + "="*60)
    print("CONVERSATION EXAMPLE")
    print("="*60)
    
    conv = ConversationManager(provider="openai", model="gpt-3.5-turbo")
    
    # Turn 1
    response1 = conv.chat("What's the capital of France?")
    print(f"User: What's the capital of France?")
    print(f"AI: {response1}")
    
    # Turn 2 (references previous context)
    response2 = conv.chat("What's its population?")
    print(f"\nUser: What's its population?")
    print(f"AI: {response2}")
    
    # Turn 3
    response3 = conv.chat("What's a famous landmark there?")
    print(f"\nUser: What's a famous landmark there?")
    print(f"AI: {response3}")


def error_handling_example():
    """Example error handling"""
    print("\n" + "="*60)
    print("ERROR HANDLING EXAMPLE")
    print("="*60)
    
    import openai
    
    try:
        client = openai.OpenAI(api_key="invalid-key")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except openai.AuthenticationError:
        print("❌ Authentication failed - check API key")
    except openai.RateLimitError:
        print("❌ Rate limit exceeded - slow down requests")
    except openai.APIError as e:
        print(f"❌ API error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("LAB 1: Minimal LLM API Usage")
    print("="*60)
    
    # Test each provider
    print("\n1. Testing OpenAI...")
    openai_example()
    
    print("\n2. Testing Anthropic...")
    anthropic_example()
    
    print("\n3. Testing Google...")
    google_example()
    
    # Conversation management
    conversation_example()
    
    # Error handling
    error_handling_example()
    
    print("\n" + "="*60)
    print("✅ Lab 1 Complete!")
    print("\nWhat to try next:")
    print("1. Change the temperature (0 to 2)")
    print("2. Try different models (gpt-4, claude-3-opus, etc.)")
    print("3. Add more conversation turns")
    print("4. Implement retry logic for failures")

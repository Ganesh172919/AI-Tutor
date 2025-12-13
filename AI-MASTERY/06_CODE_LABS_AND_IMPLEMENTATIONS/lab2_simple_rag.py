"""
Lab 2: Simple RAG (Retrieval-Augmented Generation) System

Build a system that:
1. Stores documents in a vector database
2. Retrieves relevant documents for a query
3. Uses retrieved context to answer questions
"""

import os
from typing import List, Dict, Tuple


# ============================================================================
# SIMPLE RAG SYSTEM
# ============================================================================

class SimpleRAG:
    """Minimal RAG implementation using ChromaDB"""
    
    def __init__(self, collection_name="documents"):
        """Initialize RAG system"""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector database
            self.client = chromadb.Client()
            
            # Create or get collection
            try:
                self.collection = self.client.create_collection(collection_name)
            except:
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(collection_name)
            
            print(f"✓ RAG system initialized")
            
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Run: pip install chromadb sentence-transformers")
            raise
    
    def add_documents(self, documents: List[str], ids: List[str] = None):
        """Add documents to vector database"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Embed documents
        embeddings = self.embedder.encode(documents).tolist()
        
        # Add to database
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        
        print(f"✓ Added {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant documents"""
        # Embed query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        docs = []
        for i in range(len(results['ids'][0])):
            docs.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return docs
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """Answer question using RAG"""
        # Retrieve relevant documents
        docs = self.retrieve(question, top_k)
        
        # Build context
        context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
                               for i, doc in enumerate(docs)])
        
        # Build prompt
        prompt = f"""Answer the question based on the context below. If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
        
        # Call LLM
        answer = self._call_llm(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'sources': docs
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (using OpenAI as example)"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Lower for factual answers
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            # Fallback for demo
            return f"[LLM call failed: {e}. Would normally answer based on context.]"


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================

def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Split text into chunks by sentences"""
    # Simple sentence splitting (improve with nltk/spacy)
    sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk.strip())
    
    return chunks


def chunk_by_paragraphs(text: str) -> List[str]:
    """Split text into chunks by paragraphs"""
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_by_tokens(text: str, max_tokens: int = 200) -> List[str]:
    """Split text into chunks by token count"""
    words = text.split()
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) // 4 + 1  # Rough token estimate
        
        if current_length + word_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================================
# EXAMPLES
# ============================================================================

def basic_rag_example():
    """Basic RAG example"""
    print("\n" + "="*60)
    print("BASIC RAG EXAMPLE")
    print("="*60)
    
    # Create RAG system
    rag = SimpleRAG(collection_name="ai_facts")
    
    # Add some documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        
        "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.",
        
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
        
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and generate human language."
    ]
    
    rag.add_documents(documents)
    
    # Ask questions
    questions = [
        "Who created Python?",
        "What is machine learning?",
        "How do neural networks work?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = rag.answer_question(q, top_k=2)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['content'][:100]}...")


def document_qa_example():
    """Q&A over a larger document"""
    print("\n" + "="*60)
    print("DOCUMENT Q&A EXAMPLE")
    print("="*60)
    
    # Sample article about photosynthesis
    article = """
Photosynthesis is the process by which plants convert light energy into chemical energy. 
This process occurs primarily in the leaves of plants, specifically in chloroplasts.

Chloroplasts contain chlorophyll, a green pigment that absorbs light energy from the sun. 
This light energy is used to convert carbon dioxide from the air and water from the soil into glucose, a simple sugar.

The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
This means six molecules of carbon dioxide plus six molecules of water, using light energy, 
produces one molecule of glucose and six molecules of oxygen.

Photosynthesis has two main stages: light-dependent reactions and light-independent reactions (Calvin cycle).
Light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
The Calvin cycle occurs in the stroma and uses ATP and NADPH to convert CO2 into glucose.

Photosynthesis is crucial for life on Earth. It produces oxygen that humans and animals breathe, 
and it forms the base of most food chains by converting solar energy into chemical energy stored in plants.
"""
    
    # Create RAG system
    rag = SimpleRAG(collection_name="photosynthesis")
    
    # Chunk the article
    chunks = chunk_by_sentences(article, sentences_per_chunk=2)
    print(f"Split article into {len(chunks)} chunks")
    
    # Add chunks
    rag.add_documents(chunks)
    
    # Ask questions
    questions = [
        "What is photosynthesis?",
        "Where does photosynthesis occur?",
        "What are the products of photosynthesis?",
        "What are the two stages of photosynthesis?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        result = rag.answer_question(q, top_k=2)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")


# ============================================================================
# ADVANCED: Multi-document RAG
# ============================================================================

def multi_document_example():
    """RAG over multiple documents"""
    print("\n" + "="*60)
    print("MULTI-DOCUMENT RAG EXAMPLE")
    print("="*60)
    
    rag = SimpleRAG(collection_name="tech_docs")
    
    # Simulate multiple documents
    docs = {
        "python_intro.txt": "Python is an interpreted, high-level programming language. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        
        "javascript_intro.txt": "JavaScript is a programming language commonly used in web development. It can run in web browsers and on servers using Node.js.",
        
        "java_intro.txt": "Java is a class-based, object-oriented programming language designed to be platform-independent. It runs on the Java Virtual Machine (JVM)."
    }
    
    # Add documents with metadata
    for filename, content in docs.items():
        chunks = chunk_by_sentences(content, sentences_per_chunk=1)
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        rag.add_documents(chunks, ids)
    
    # Query
    result = rag.answer_question("Which languages can run in web browsers?")
    print(f"Q: Which languages can run in web browsers?")
    print(f"A: {result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['id']}: {source['content']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("LAB 2: Simple RAG System")
    print("="*60)
    
    # Run examples
    basic_rag_example()
    document_qa_example()
    multi_document_example()
    
    print("\n" + "="*60)
    print("✅ Lab 2 Complete!")
    print("\nWhat to try next:")
    print("1. Add your own documents")
    print("2. Try different chunk sizes")
    print("3. Experiment with top_k values")
    print("4. Add metadata to documents")
    print("5. Implement hybrid search (keyword + semantic)")

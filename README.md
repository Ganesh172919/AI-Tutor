# 🎓 AI Tutor

**Learn anything with simple explanations and real-life examples.**

AI Tutor is a production-ready tutoring system powered by Google's Gemini API. It teaches any subject using adaptive micro-lessons, simple English, and personalized analogies.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI TUTOR ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐     ┌─────────────────────────────────────────────────────┐  │
│   │ Learner │────▶│                    Frontend (React)                 │  │
│   └─────────┘     │  • Onboarding Quiz   • Lesson View   • Chat UI     │  │
│                   └────────────────────────────┬────────────────────────┘  │
│                                                │                            │
│                                                ▼                            │
│                   ┌─────────────────────────────────────────────────────┐  │
│                   │                   FastAPI Backend                    │  │
│                   │  • REST Endpoints  • Session Management  • Auth     │  │
│                   └───────────┬─────────────────────┬───────────────────┘  │
│                               │                     │                       │
│              ┌────────────────┴───────┐   ┌────────┴────────┐              │
│              ▼                        ▼   ▼                  ▼              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐     │
│   │  Tutor Engine   │   │  Prompt Builder │   │    Memory Store     │     │
│   │ • Onboarding    │   │ • Templates     │   │ • Learner Profiles  │     │
│   │ • Explanations  │   │ • Safe Filling  │   │ • Progress Tracking │     │
│   │ • Evaluation    │   │ • Injection     │   │ • Analytics         │     │
│   │ • Hints         │   │   Prevention    │   │ • Export/Delete     │     │
│   └────────┬────────┘   └─────────────────┘   └─────────────────────┘     │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        Gemini Client                                 │  │
│   │  • Model Selection (Fast/Balanced/Deep)  • Retry Logic              │  │
│   │  • Circuit Breaker  • Rate Limiting  • Analytics Logging            │  │
│   └─────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                           │
│                                 ▼                                           │
│                   ┌─────────────────────────┐                              │
│                   │     Gemini API          │                              │
│                   │  (Google AI Studio)     │                              │
│                   └─────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **🎯 Adaptive Assessment** - Quick 3-7 question quiz to gauge your level
- **📚 Micro-Lessons** - 2-8 minute bite-sized lessons with clear objectives
- **🗣️ Simple English** - No jargon. Short sentences. Easy to understand.
- **🍳 Real-Life Analogies** - Explanations using cooking, sports, money, or your choice
- **✏️ Worked Examples** - Step-by-step solutions with actual numbers
- **💡 Smart Hints** - Progressive hints that guide without giving answers
- **📊 Progress Tracking** - See what you've mastered and what needs review
- **🔄 Spaced Repetition** - Reminders to review topics for retention
- **🔒 Privacy First** - Export or delete your data anytime

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/AI-Tutor.git
cd AI-Tutor

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"

# Windows (CMD)
set GEMINI_API_KEY=your-api-key-here

# Mac/Linux
export GEMINI_API_KEY=your-api-key-here
```

### 3. Run the Backend

```bash
cd src/backend
python -m uvicorn main:app --reload
```

The API is now running at http://localhost:8000

### 4. Open the Frontend

Simply open `src/frontend/index.html` in your browser, or serve it:

```bash
# Python's built-in server
cd src/frontend
python -m http.server 3000
```

Then visit http://localhost:3000

---

## 📖 How It Works

### 1. Onboarding

When you start, the tutor:
1. Asks what you want to learn
2. Asks how you prefer analogies (cooking, sports, everyday life)
3. Gives a short quiz (3-7 questions)
4. Creates your learner profile

### 2. Lessons

Each micro-lesson includes:
- **TL;DR**: One sentence summary
- **Explanation**: 2-3 sentences in simple English
- **Analogy**: Real-life example from your preferred domain
- **Worked Example**: Step-by-step solution with numbers
- **Practice Question**: Test your understanding
- **Hint System**: Get help without getting the answer

### 3. Progress

The system tracks:
- Topics you've mastered (80%+ correct after 3+ attempts)
- Topics in progress
- Common mistakes to watch for
- When to review for retention

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | Required |
| `GEMINI_FAST_MODEL` | Model for quick hints | `gemini-1.5-flash` |
| `GEMINI_BALANCED_MODEL` | Model for explanations | `gemini-1.5-flash` |
| `GEMINI_DEEP_MODEL` | Model for lesson planning | `gemini-1.5-pro` |
| `GEMINI_MAX_RETRIES` | API retry attempts | `3` |
| `GEMINI_TIMEOUT` | Request timeout (seconds) | `30` |

### Model Tiers

The system uses different models for different tasks:

- **FAST** (`gemini-1.5-flash`): Hints, quick feedback
- **BALANCED** (`gemini-1.5-flash`): Explanations, evaluations
- **DEEP** (`gemini-1.5-pro`): Lesson planning, complex topics

This balances quality and cost.

---

## 📁 Project Structure

```
AI-Tutor/
├── src/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   ├── gemini_client.py  # Gemini API wrapper
│   │   ├── prompt_builder.py # Template management
│   │   ├── memory_store.py   # Data persistence
│   │   └── tutor_engine.py   # Core tutoring logic
│   └── frontend/
│       └── index.html        # React-based UI
├── prompts/                   # Prompt templates
│   ├── system_tutor.txt
│   ├── onboarding_quiz.txt
│   ├── explain_concept.txt
│   └── ...
├── examples/                  # Sample data
│   ├── pythagorean_theorem_prompt.json
│   ├── for_loop_prompt.json
│   ├── compound_interest_prompt.json
│   └── sample_transcript_*.json
├── tests/                     # Test suite
│   ├── test_gemini_client.py
│   ├── test_prompt_builder.py
│   ├── test_memory_store.py
│   └── test_tutor_engine.py
├── deploy/                    # Deployment configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── CLOUD_DEPLOY.md
├── requirements.txt
└── README.md
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gemini_client.py -v

# Run with coverage
pytest tests/ --cov=src/backend --cov-report=html
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
cd deploy

# Set your API key
export GEMINI_API_KEY=your-key-here

# Start everything
docker-compose up -d

# View logs
docker-compose logs -f
```

Services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Production Deployment

See [deploy/CLOUD_DEPLOY.md](deploy/CLOUD_DEPLOY.md) for instructions on:
- Google Cloud Run (recommended)
- AWS App Runner / ECS
- Azure Container Apps
- DigitalOcean App Platform
- Simple VPS setup

---

## 📡 API Endpoints

### Onboarding

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/onboarding/start` | Start onboarding, get quiz questions |
| POST | `/api/onboarding/complete` | Submit quiz answers, get profile |

### Lessons

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/lesson/plan` | Create micro-lesson plan |
| POST | `/api/lesson/explain` | Get concept explanation |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluate/answer` | Check answer, get feedback |
| POST | `/api/evaluate/hint` | Get hint for question |
| POST | `/api/evaluate/summary` | Get lesson summary |

### Progress

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/progress/{learner_id}` | Get overall progress |
| GET | `/api/progress/{learner_id}/review` | Get topics due for review |
| PUT | `/api/profile/{learner_id}` | Update preferences |

### Privacy

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/{learner_id}/export` | Export all user data |
| DELETE | `/api/data/{learner_id}` | Delete all user data |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Free-form tutoring chat |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/summary` | Aggregate (anonymized) stats |
| GET | `/api/analytics/gemini` | API usage statistics |

---

## 🎨 Customization

### Adding New Prompt Templates

1. Create a `.txt` file in `prompts/`:
   ```
   prompts/my_new_template.txt
   ```

2. Use `{{placeholders}}` for variables:
   ```
   Explain {{topic}} to a {{level}} learner.
   Use analogies about {{analogy_domain}}.
   ```

3. Use in code:
   ```python
   from prompt_builder import build_prompt
   
   prompt = build_prompt(
       "my_new_template",
       topic="fractions",
       level="beginner",
       analogy_domain="cooking"
   )
   ```

### Extending the Memory Store

To use PostgreSQL instead of JSON files:

1. Install: `pip install asyncpg sqlalchemy`
2. See `memory_store.py` for the PostgreSQL migration guide
3. Update the `MemoryStore` class with database calls

---

## 📊 Sample Teaching Output

Here's how the tutor explains the Pythagorean theorem:

```
TL;DR: In a right triangle, the longest side squared equals the 
sum of the other two sides squared.

Explanation: The Pythagorean theorem is a formula for right triangles. 
A right triangle has one 90-degree corner, like the corner of a book. 
The formula says: a² + b² = c², where c is the longest side.

Think of it like this: Imagine you have two small square rugs. 
If you combine their areas, you get the area of one bigger square rug. 
The sides of those rugs are the sides of your triangle!

Worked Example:
Problem: Find the longest side of a triangle with sides 3 and 4.

Step 1: Write the formula → a² + b² = c²
Step 2: Put in numbers → 3² + 4² = c²
Step 3: Calculate → 9 + 16 = c²
Step 4: Add → 25 = c²
Step 5: Square root → c = √25 = 5

Practice Question: A triangle has sides 5 and 12. Find the longest side.

Hint: Square both numbers, add them, then take the square root.
```

---

## 🔒 Privacy & Security

- **No conversation storage** by default - only compact learner state
- **Export your data** anytime via `/api/data/{id}/export`
- **Delete your data** anytime via `DELETE /api/data/{id}`
- **Anonymized analytics** - personal info is hashed
- **Input sanitization** - prompts are escaped to prevent injection

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built with [Gemini API](https://ai.google.dev/)
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- Frontend uses [React](https://react.dev/)

---

## 📧 Support

- Open an issue for bugs or features
- Check existing issues before creating new ones
- Include logs and steps to reproduce for bugs

---

**Happy Learning! 🎉**
production-ready AI Tutor system that teaches any topic with the fastest possible understanding for the user

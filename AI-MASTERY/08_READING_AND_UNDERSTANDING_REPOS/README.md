# 08 – How to Read Any AI Repo Fast

## 🎯 What You'll Learn

* How to quickly understand any AI codebase
* Where to look first
* What patterns to recognize
* How to navigate large repos efficiently

---

## 🗺️ The Mental Model

> **AI repos tell a story: Data → Model → Training → Inference**

Every AI repository follows this flow (sometimes hidden):

```
Input Data → Processing → Model Definition → Training Loop → Inference/Deployment
```

Your job: Find each piece and understand how they connect.

---

## 📂 Where to Look First

### 1. README.md (Always Start Here)

**What to look for:**
* What does this project do?
* How to install?
* Quick start example
* Architecture diagram (if provided)

**Time: 2-3 minutes**

### 2. Requirements / Dependencies

**Files to check:**
* `requirements.txt` (Python)
* `package.json` (Node.js)
* `Cargo.toml` (Rust)
* `go.mod` (Go)

**What it tells you:**
* Main framework (PyTorch, TensorFlow, JAX, LangChain)
* Whether it's inference-only or includes training
* External APIs used (OpenAI, Anthropic, etc.)

**Time: 1 minute**

### 3. Project Structure

**Look at directory layout:**

```bash
ls -la
tree -L 2  # if tree is installed
```

**Common patterns:**

```
AI-Project/
├── data/           # Training/test data
├── models/         # Model definitions
├── training/       # Training scripts
├── inference/      # Inference/prediction
├── configs/        # Configuration files
├── notebooks/      # Jupyter notebooks
├── tests/          # Unit tests
└── scripts/        # Utility scripts
```

**What to notice:**
* Is there a `data/` folder? → Includes dataset handling
* Is there a `training/` folder? → Includes training code
* Is there a `notebooks/` folder? → Exploration/experiments
* Multiple `config` files → Configurable system

**Time: 2 minutes**

---

## 🔍 Reading Strategy

### Phase 1: High-Level Understanding (10 minutes)

**Goal:** Understand what the project does

1. **Read README** (3 min)
2. **Skim main entry point** (3 min)
   * `main.py`, `app.py`, `train.py`, `inference.py`
   * Look for high-level flow, not details
3. **Check configs** (2 min)
   * What parameters are configurable?
   * What models/datasets are referenced?
4. **Look at examples** (2 min)
   * `examples/` folder
   * Usage in README

**Output:** You can now explain the project in 2 sentences.

### Phase 2: Core Components (20 minutes)

**Goal:** Understand key pieces

1. **Model architecture** (10 min)
   * Find model definition file
   * Understand layers/components
   * Note input/output shapes

2. **Data processing** (5 min)
   * How is data loaded?
   * What preprocessing happens?
   * What format is expected?

3. **Main logic** (5 min)
   * Training loop (if applicable)
   * Inference pipeline
   * How pieces connect

**Output:** You can modify configurations and run the code.

### Phase 3: Deep Dive (60+ minutes)

**Goal:** Understand implementation details

1. **Read core functions** line-by-line
2. **Trace data flow** through the system
3. **Understand design decisions**
4. **Read tests** (reveals intended behavior)

**Output:** You can contribute code or debug issues.

---

## 🧩 Common Patterns to Recognize

### Pattern 1: LLM Wrapper / API Client

**Structure:**
```
project/
├── client.py      # API wrapper
├── prompts.py     # Prompt templates
├── main.py        # Usage example
└── config.py      # API keys, settings
```

**How to read:**
1. Check `client.py` for API provider (OpenAI, Anthropic, etc.)
2. Look at `prompts.py` for task-specific prompts
3. See `main.py` for usage patterns

**Key files:** `client.py`, `prompts.py`

---

### Pattern 2: RAG System

**Structure:**
```
project/
├── embeddings.py      # Embedding logic
├── vector_store.py    # DB operations
├── retriever.py       # Search logic
├── generator.py       # LLM generation
└── pipeline.py        # Combine retrieve + generate
```

**How to read:**
1. Start with `pipeline.py` (orchestration)
2. Understand `retriever.py` (how docs are found)
3. Check `vector_store.py` (what DB is used)
4. See `generator.py` (how LLM is called)

**Key files:** `pipeline.py`, `retriever.py`

---

### Pattern 3: Agent System

**Structure:**
```
project/
├── agents/
│   ├── planner.py
│   ├── executor.py
│   └── verifier.py
├── tools/
│   ├── search.py
│   ├── calculator.py
│   └── code_runner.py
├── memory.py
└── orchestrator.py
```

**How to read:**
1. Start with `orchestrator.py` (how agents work together)
2. Read one agent file (`planner.py`)
3. Check `tools/` to see capabilities
4. Understand `memory.py` (state management)

**Key files:** `orchestrator.py`, `agents/planner.py`

---

### Pattern 4: Fine-Tuning Project

**Structure:**
```
project/
├── data/
│   ├── prepare.py     # Data preprocessing
│   └── dataset.py     # PyTorch Dataset
├── models/
│   └── model.py       # Model architecture
├── train.py           # Training script
├── configs/
│   └── config.yaml    # Hyperparameters
└── evaluate.py        # Evaluation
```

**How to read:**
1. Check `configs/config.yaml` (hyperparameters)
2. Read `data/prepare.py` (data format)
3. Understand `models/model.py` (architecture)
4. Trace `train.py` (training loop)

**Key files:** `train.py`, `models/model.py`, `configs/config.yaml`

---

### Pattern 5: Transformer Implementation

**Structure:**
```
project/
├── attention.py       # Attention mechanism
├── transformer.py     # Full transformer
├── tokenizer.py       # Tokenization
├── train.py           # Training
└── generate.py        # Text generation
```

**How to read:**
1. Start with `transformer.py` (overall architecture)
2. Read `attention.py` (core mechanism)
3. Check `generate.py` (inference)
4. Optionally: `train.py` (training details)

**Key files:** `transformer.py`, `attention.py`

---

## 🎯 Specific Examples

### Example 1: LangChain Repository

**Quick scan:**
```
langchain/
├── chains/           # Pre-built chains
├── agents/           # Agent implementations
├── llms/             # LLM integrations
├── prompts/          # Prompt templates
├── vectorstores/     # Vector DB integrations
└── memory/           # Conversation memory
```

**Reading strategy:**
1. Pick one use case (e.g., "build RAG")
2. Find example in `examples/`
3. Trace imports to relevant modules
4. Read only those modules

**Key insight:** LangChain is modular—you don't need to understand everything, just the parts you're using.

---

### Example 2: nanoGPT (Minimal GPT)

**Quick scan:**
```
nanoGPT/
├── model.py          # GPT model (250 lines)
├── train.py          # Training script
├── sample.py         # Text generation
└── config/           # Configs for different sizes
```

**Reading strategy:**
1. Read `model.py` top to bottom (it's small!)
2. Understand `CausalSelfAttention` class
3. See how `train.py` uses the model
4. Try `sample.py` to generate text

**Key insight:** Clean, minimal implementation—great for learning.

---

### Example 3: llama.cpp

**Quick scan:**
```
llama.cpp/
├── llama.h           # C++ header
├── llama.cpp         # Core implementation
├── examples/
│   ├── main/         # CLI inference
│   └── server/       # API server
└── convert.py        # Convert PyTorch models
```

**Reading strategy:**
1. Understand it's C++ (not Python)
2. Check `examples/main/` for usage
3. Read `llama.h` for API
4. Only dive into `llama.cpp` if needed

**Key insight:** Focused on efficient inference, not training.

---

## 🔧 Tools to Help

### 1. Code Search

**Use `grep` or `ripgrep`:**
```bash
# Find where a function is defined
rg "def train_model"

# Find all LLM API calls
rg "openai.ChatCompletion"

# Find config loading
rg "load_config|config.yaml"
```

### 2. Dependency Graph

**Visualize imports:**
```bash
# Python
pydeps project/ --max-depth 2

# Or use your IDE's "Go to Definition" feature
```

### 3. Git History

**Understand evolution:**
```bash
# See recent changes
git log --oneline -20

# See file history
git log --follow model.py

# See what changed
git show <commit-hash>
```

### 4. IDE Features

**Use your IDE:**
* **Go to Definition** (Ctrl+Click)
* **Find References** (where is this used?)
* **Call Hierarchy** (what calls this?)
* **Type Hints** (what's the expected type?)

---

## 📝 Reading Checklist

When exploring a new repo:

**Quick Pass (10 min):**
- [ ] Read README
- [ ] Check requirements/dependencies
- [ ] Understand directory structure
- [ ] Find main entry point
- [ ] Run example (if possible)

**Deep Pass (30 min):**
- [ ] Identify model/architecture
- [ ] Understand data flow
- [ ] Read configuration options
- [ ] Trace one complete workflow
- [ ] Check tests

**Expert Pass (1-2 hours):**
- [ ] Read core implementation files
- [ ] Understand design decisions
- [ ] Identify optimization opportunities
- [ ] Check for limitations/issues
- [ ] Try modifying code

---

## 💡 Pro Tips

1. **Don't read linearly** - Jump to interesting parts
2. **Run before reading** - See it work first
3. **Start from examples** - Work backwards to implementation
4. **Use debugger** - Step through code execution
5. **Draw diagrams** - Sketch data flow
6. **Ask the repo** - Many projects have Discord/forums
7. **Read tests** - They show intended behavior
8. **Check issues/PRs** - Common problems and solutions

---

## 🤔 Quick Quiz

1. What's the first file you should read in any repo?
2. How can you tell if a repo includes training code?
3. Name three common patterns in AI repos.
4. What tool helps you search for function definitions?

---

**Next:** [09 - Key Terminology Glossary →](../09_KEY_TERMINOLOGY_GLOSSARY/README.md)

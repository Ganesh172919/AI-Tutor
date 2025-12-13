# 04 – Building Models From Scratch

## 🎯 What You'll Learn

* Conceptual understanding of model training
* Data → Tokens → Model → Loss → Updates
* Why training is expensive
* How to train a tiny transformer
* What gradients actually mean
* Why overfitting happens
* Fine-tuning vs training from scratch

---

## 🔄 The Training Pipeline

### The Big Picture

```
Raw Data → Preprocessing → Tokenization → Training → Evaluation → Deployment
```

Let's break down each step.

---

## 📊 Step 1: Data Collection

### What You Need

**For pretraining:**
* Massive text corpus (100GB - 10TB+)
* Diverse sources: books, web, code, conversations
* Cleaned and deduplicated

**For fine-tuning:**
* Much smaller (1MB - 1GB)
* Task-specific examples
* High quality > quantity

### Common Datasets

| Dataset | Size | Content |
|---------|------|---------|
| Common Crawl | 250TB+ | Web pages |
| The Pile | 800GB | Books, papers, code, Wikipedia |
| Wikipedia | 20GB | Encyclopedia articles |
| GitHub | 1TB+ | Source code |
| RedPajama | 1.2TB | Open reproduction of LLaMA data |

### Data Quality Matters

**Good training data:**
```
"Photosynthesis is the process by which plants convert 
sunlight into chemical energy using chlorophyll."
```

**Bad training data:**
```
"BUY N0W!!! Click here!!! Amazing deal!!!"
```

Models learn from everything—garbage in, garbage out.

---

## 🔢 Step 2: Tokenization

Already covered in [Section 01](../01_HOW_LLMs_WORK_DEEPLY/README.md), but quick recap:

```python
text = "Hello world"
tokens = tokenizer.encode(text)  # [15496, 995]
```

Each token gets an ID (0-50000 typically).

---

## 🧱 Step 3: Model Architecture

### Simplified Transformer

```
Input: Token IDs [15496, 995]
       ↓
Embedding Layer: IDs → Vectors
       [15496] → [0.2, -0.5, 0.1, ...]  (512 dims)
       [995]   → [0.1, 0.3, -0.2, ...]
       ↓
Transformer Blocks (12-96 layers)
   Each block:
   1. Self-Attention (tokens look at each other)
   2. Feed-Forward (process each token)
   3. Layer Norm (stabilize training)
       ↓
Output Layer: Vectors → Token Probabilities
       [0.2, -0.5, ...] → [0.001, 0.003, ..., 0.12, ...]
                           ↑
                        50,000 probabilities (one per token)
```

### Key Parameters

| Component | Parameters |
|-----------|------------|
| Embedding | vocab_size × hidden_dim |
| Attention | 4 × hidden_dim² (per layer) |
| Feed-Forward | 8 × hidden_dim² (per layer) |
| Output Layer | vocab_size × hidden_dim |

**Example (GPT-2 Small):**
* Vocab: 50,000
* Hidden dim: 768
* Layers: 12
* **Total parameters: 117M**

---

## 📉 Step 4: Loss Function

### What is Loss?

**Loss = How wrong the model is**

Lower loss = better predictions

### Next-Token Prediction Loss

For each position, compare:
* **Model's prediction:** [0.001, 0.003, ..., 0.12, ...]
* **Actual next token:** Token #995

**Cross-Entropy Loss:**
```
Loss = -log(probability of correct token)

If model gave 0.8 to correct token:
  Loss = -log(0.8) = 0.22 (good!)

If model gave 0.01 to correct token:
  Loss = -log(0.01) = 4.6 (bad!)
```

### Average Loss

Train on billions of examples, average the loss.

**Good loss values:**
* Random model: ~10.0 (just guessing)
* Trained model: ~2.0-3.0
* State-of-the-art: ~1.5-2.0

---

## 🔧 Step 5: Optimization (Training)

### The Goal

**Adjust model parameters to minimize loss**

### How: Gradient Descent

1. **Forward Pass:** Input → Model → Predictions → Loss
2. **Backward Pass:** Compute gradients (how to change weights)
3. **Update:** Adjust weights slightly in direction of improvement

```
weight_new = weight_old - learning_rate × gradient
```

### What Are Gradients?

**Simple explanation:** Gradients tell you which direction to move weights to reduce loss.

**Analogy:** You're in fog on a hill. Gradient tells you which way is downhill.

**Math (don't worry if confusing):**
```
gradient = ∂Loss/∂weight

"How much does loss change if I change this weight?"
```

### Learning Rate

**Too high:** Model doesn't converge (overshoots)
**Too low:** Training takes forever
**Just right:** 10⁻⁴ to 10⁻⁶ typically

### Batching

Train on 32-512 examples at once (batch), then update.

**Why?**
* Efficient on GPUs
* Smoother updates
* Faster training

---

## 💰 Step 6: Why Training is Expensive

### Compute Cost

**GPT-3 (175B parameters):**
* Training time: ~1 month
* Hardware: 10,000 GPUs
* Cost: ~$5-10 million

**Why so expensive?**

1. **Trillions of tokens:** 300B+ tokens processed
2. **Massive parameters:** 175B weights to update
3. **Many iterations:** Multiple passes through data
4. **High-end hardware:** A100/H100 GPUs ($10K-40K each)

### Energy Cost

Training GPT-3 ≈ 1,300 MWh ≈ 500 tons of CO₂

(Roughly equivalent to driving a car for 1 million miles)

### Economic Reality

**Pretraining from scratch:** Only feasible for big companies

**Fine-tuning:** Affordable for individuals/startups ($100-10K)

---

## 🎨 Step 7: Fine-Tuning

### What is Fine-Tuning?

**Take a pretrained model + train it a bit more on specific data**

```
Base Model (trained on internet) → +Fine-tuning (on medical texts) → Medical AI
```

### Why Fine-Tune?

1. **Cheaper:** Don't need to train from scratch
2. **Faster:** Hours instead of months
3. **Better:** Leverages general knowledge + specializes
4. **Less data:** Can work with 1,000-10,000 examples

### Fine-Tuning Process

```python
# Simplified
model = load_pretrained("gpt-3.5-turbo")
training_data = load_data("medical_qa.jsonl")

model.train(
    data=training_data,
    epochs=3,
    learning_rate=1e-5,  # Much smaller than pretraining
    batch_size=16
)

model.save("medical-gpt-3.5")
```

### Common Fine-Tuning Types

**1. Instruction Fine-Tuning**
```json
{"instruction": "Explain photosynthesis", "response": "..."}
{"instruction": "Write a poem about cats", "response": "..."}
```

**2. Task-Specific Fine-Tuning**
```json
{"input": "This movie was great!", "output": "positive"}
{"input": "Terrible product.", "output": "negative"}
```

**3. Conversational Fine-Tuning**
```json
{"user": "Hello", "assistant": "Hi! How can I help?"}
{"user": "What's the weather?", "assistant": "I don't have real-time data..."}
```

---

## 🧪 Code Example: Tiny Transformer

See `tiny_transformer.py` for a minimal implementation.

**What it demonstrates:**
* Token embeddings
* Self-attention (simplified)
* Feed-forward layers
* Training loop
* Loss calculation

---

## 🔍 Understanding Overfitting

### What is Overfitting?

**Model memorizes training data instead of learning patterns**

### Example

**Training data:**
```
"The cat sat on the mat" → common
"The dog ran in the park" → common
```

**Overfitted model:**
```
Input: "The cat sat on the"
Output: "mat" (correct, but only because memorized)

Input: "The cat sat on the table"
Output: "mat" (wrong! Just repeating training data)
```

### Signs of Overfitting

* **Training loss:** Goes down ✓
* **Validation loss:** Goes up ✗

```
Epoch 1: train=3.0, val=3.1
Epoch 5: train=2.0, val=2.2
Epoch 10: train=1.0, val=2.5  ← Overfitting starts
Epoch 20: train=0.5, val=3.0  ← Severe overfitting
```

### How to Prevent

1. **More data:** Harder to memorize
2. **Regularization:** Penalize complex models
3. **Dropout:** Randomly disable neurons during training
4. **Early stopping:** Stop when validation loss increases
5. **Data augmentation:** Create variations of examples

---

## 📊 Training vs Fine-Tuning Comparison

| Aspect | Pretraining | Fine-Tuning |
|--------|-------------|-------------|
| Data needed | 100GB - 10TB | 1MB - 1GB |
| Time | Weeks-months | Hours-days |
| Cost | $1M - $100M | $100 - $10K |
| Hardware | 1,000s of GPUs | 1-8 GPUs |
| Purpose | Learn general knowledge | Specialize |
| Who does it | Big companies | Anyone |

---

## 🛠️ Practical Guide: Fine-Tune Your Own Model

### Option 1: OpenAI Fine-Tuning

```python
import openai

# Prepare data
training_file = openai.File.create(
    file=open("training_data.jsonl", "rb"),
    purpose='fine-tune'
)

# Start fine-tuning
fine_tune = openai.FineTune.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo"
)

# Cost: ~$0.008 per 1K tokens
# Time: 1-2 hours for small dataset
```

### Option 2: Hugging Face

```python
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load your data
train_dataset = load_dataset("your_data.json")

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args
)

trainer.train()
```

### Option 3: LoRA (Efficient Fine-Tuning)

**LoRA = Low-Rank Adaptation**

Instead of updating all weights, update small "adapter" layers.

**Benefits:**
* 100x less storage
* 3x faster training
* Same quality as full fine-tuning

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(base_model, config)
# Now train as normal
```

---

## 🤔 Quick Quiz

1. What are the three main stages of training a language model?
2. What is a loss function?
3. What do gradients tell you during training?
4. Why is fine-tuning cheaper than training from scratch?
5. What is overfitting and how do you detect it?

---

**Next:** [05 - Practical LLM Engineering →](../05_PRACTICAL_LLM_ENGINEERING/README.md)

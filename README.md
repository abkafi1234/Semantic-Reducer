# Semantic Reducer

A high-performance Python library that bridges the gap between **Deep Learning** and **traditional Machine Learning**.

`semantic-reducer` uses **BERT contextual embeddings** and **FAISS vector search** to semantically normalize and reduce the vocabulary of a text corpus.

By mapping semantically equivalent words (for example: *fast*, *quick*, *rapid*) to a single, highly frequent canonical word, it creates cleaner and denser input for traditional ML models such as **TF-IDF + SVM**, **Random Forest**, or **Logistic Regression**.

After the initial processing stage, inference on new text requires only a dictionary lookup, enabling **O(1) runtime complexity**.

Example : https://colab.research.google.com/drive/1qZAxCif_shYCu1gIfUtU-zmDLVc4tpEm?usp=sharing

---

# Features

- **Context-Aware Processing**  
  Uses Transformer models (default: `bert-base-multilingual-cased`) to capture contextual meaning instead of relying on static embeddings.

- **Smart Vocabulary Reduction**  
  Automatically replaces rare or obscure words with their most frequent semantic equivalents.

- **O(1) Inference Speed**  
  After training, the BERT and FAISS pipeline is no longer required. Text reduction happens via a lightweight Python dictionary.

- **Production Ready**  
  Supports saving and loading compiled reduction maps for deployment.

- **Hardware Agnostic**  
  Automatically detects and uses GPU (CUDA) if available, while remaining fully functional on CPU.

---

# Installation

Install via pip:

```bash
pip install semantic-reducer
```

---

# Quick Start

## 1. Training and Building the Reduction Map

```python
from semantic_reducer import SemanticReducer

# Initialize the reducer
# (downloads the BERT model on first run)
reducer = SemanticReducer()

# Training corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast dark-colored canine leaps above a tired hound.",
    "Speedy foxes jump over sleepy dogs."
]

# Extract contextual representations
reducer.process_corpus_contextually(corpus, batch_size=2)

# Finalize embeddings and create FAISS index
reducer.finalize_embeddings()
reducer.build_index()

# Build the semantic reduction dictionary
reducer.build_reduction_map(
    threshold=0.85,   # similarity threshold
    top_k=5           # number of neighbors to consider
)

# Save the trained system
reducer.save_system(prefix="my_corpus")
```

---

## 2. Lightning-Fast Inference

Once the reduction map is built, **BERT is no longer required**. New text is processed through a dictionary lookup.

```python
from semantic_reducer import SemanticReducer

# Initialize reducer
reducer = SemanticReducer()

# Load saved system
reducer.load_system(prefix="my_corpus")

# Reduce new text instantly
new_text = "The rapid fox leaped."
reduced_text = reducer.reduce_text(new_text)

print(f"Original: {new_text}")
print(f"Reduced:  {reduced_text}")
```

---

# Workflow Overview

1. **Contextual Encoding**  
   Extract contextual word embeddings using BERT.

2. **Vector Indexing**  
   Store embeddings in a FAISS similarity search index.

3. **Semantic Clustering**  
   Identify semantically similar words.

4. **Canonical Replacement**  
   Replace each cluster with the most frequent word in the corpus.

5. **Dictionary Compilation**  
   Store the final mappings for fast inference.

---

# Requirements

- Python ≥ 3.8
- torch
- numpy
- faiss-cpu
- transformers
- tqdm

Install dependencies manually if needed:

```bash
pip install torch numpy faiss-cpu transformers tqdm
```

---

# License

This project is licensed under the **MIT License**.

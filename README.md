##  Zero-Shot Color Detection using DistilBERT (MNLI)

###  Overview
This project applies **zero-shot learning** to semantic color detection in text using a **Transformer-based NLI model** — `typeform/distilbert-base-uncased-mnli`.  
Unlike keyword-based methods, it infers contextual meanings to detect both **literal** (e.g., *“The car is red”*) and **figurative** (e.g., *“I’m feeling blue”*) color references.

---

###  Key Concepts in ML & NLP

| **Technique** | **Description** |
|----------------|-----------------|
| **Zero-Shot Learning** | Classify unseen categories using pretrained models without additional training |
| **Transfer Learning** | Reuse an NLI model (entailment/contradiction) for text classification tasks |
| **Transformer Architecture** | Self-attention mechanism for modeling semantic relationships |
| **Tokenization & Embedding Extraction** | Subword tokenization and contextual embeddings from DistilBERT |
| **Threshold-Based Decisioning** | Adjust confidence thresholds to balance precision and recall |

---

###  Model Information
- **Base Model:** `typeform/distilbert-base-uncased-mnli`  
- **Task:** Natural Language Inference (Premise → Hypothesis Classification)  
- **Framework:** Hugging Face Transformers  
- **Pipeline:** `zero-shot-classification`

---

###  Model Inputs
Each input text is compared against color labels formulated as hypotheses, e.g.  
> “This text is about the color blue.”

---

###  Model Outputs
For each candidate label, the model produces:
- **Confidence score** (entailment probability)  
- **Ranked label list** sorted by likelihood

---

##  Core Pipeline

### 1️. Load Model & Tokenizer
  ``` python
from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=0 if torch.cuda.is_available() else -1
  )
``` 
### 2️. Define Color Candidates
A predefined label set of 32 color terms for detection:
 ``` python
OLOR_LABELS = ["red", "blue", "green", "yellow", "purple", "black", "white", ...]
```
---

### 3️. Color Detection Function

Performs multi-label inference using configurable confidence thresholds:
 ``` python
def detect_colors(text, confidence_threshold=0.5):
    result = classifier(text, candidate_labels=COLOR_LABELS, multi_label=True)
    return [
        (label, score)
        for label, score in zip(result['labels'], result['scores'])
        if score >= confidence_threshold
    ]
```
---
##  Comparison: Semantic vs Literal Detection

| **Method** | **Description** |
|-------------|-----------------|
| **Literal Detection** | Simple keyword matching (`color in text.lower()`) |
| **Zero-Shot Detection** | Semantic inference using NLI embeddings |

**Example:**
> **Text:** “I’m feeling blue today.”  
> **Zero-Shot:** `['blue']`  
> **Literal:** `[]`  
 *Semantic understanding succeeds even without a literal color mention.*

---

##  Threshold Sensitivity Analysis

The **confidence threshold** controls precision–recall trade-offs:

| **Threshold** | **Detection Behavior** | **Notes** |
|----------------|------------------------|------------|
| **0.3** | Many colors | High recall, lower precision |
| **0.5** | Balanced | Default trade-off |
| **0.7** | Fewer colors | High precision |
| **0.9** | Only top-confidence colors | Very selective |

---

**Example:**
> **Text:** “The sky is blue and the grass is green with yellow flowers.”  
> **Expected:** blue, green, yellow  
> **Threshold 0.5 →** Detected all 3  
> **Threshold 0.9 →** Missed *yellow*

##  Tokenization & Embeddings (Deep Dive)

###  Tokenization
Text is decomposed into **subword tokens** using the WordPiece tokenizer:

```python
['[CLS]', 'the', 'sky', 'is', 'blue', 'and', 'the', 'grass', 'is', 'green', '.', '[SEP]']
```
Embeddings:

Each token is converted into a 768-dimensional contextual vector:
```python
Embedding Tensor Shape: [1, 12, 768]
- Batch size: 1
- Sequence length: 12
- Hidden size: 768
``` 

## 🧪 Examples

### Example 1
**Input:**  
> "The sky is blue and the grass is green."

**Detected:**

| **Color** | **Confidence** | **Note** |
|------------|----------------|-----------|
| **BLUE** | 0.9821 | Very High |
| **GREEN** | 0.9543 | Very High |

---

### Example 2
**Input:**  
> "She wore a red dress with yellow flowers and carried a purple purse."

**Detected:**

| **Color** | **Confidence** | **Note** |
|------------|----------------|-----------|
| **RED** | 0.9981 | Very High |
| **YELLOW** | 0.9910 | Very High |
| **PURPLE** | 0.9834 | Very High |

## 🔬 Model Interpretation

| **Layer** | **Component** | **Role** |
|------------|----------------|-----------|
| **Embedding** | WordPiece tokenizer + positional encoding | Encodes sequence context |
| **Encoder** | Multi-head self-attention | Captures token interdependencies |
| **Pooler** | `[CLS]` token representation | Summarizes sequence semantics |
| **Classifier** | NLI head (entailment / neutral / contradiction) | Infers “is about” relationships |

---

🧑‍💻 Author

` Mourad Sleem ` 

`NLP & Deep Learning Practitioner`

`📧 mouradbshina@gmail.com`


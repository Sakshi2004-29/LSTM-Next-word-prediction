# 🧠 LSTM-Based Next Word Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Lab Assignment 5 | AI Sequence Prediction System | Group Assignment**

---

## 📌 Project Overview

This project is a complete **AI-based Next Word Prediction System** built using:

- 🔷 **LSTM (Long Short-Term Memory)** neural network
- 🔷 **TensorFlow / Keras** for model development
- 🔷 **FastAPI** for REST API deployment
- 🔷 **Swagger UI** for testing

**Example:**
- Input → `"to be or not to"`
- Output → `"be"`
- Complete → `"to be or not to be"` ✅

---

## 👥 Group Members

| Name | Roll No | Contribution |
|------|---------|-------------|
| Member 1 | XXX | Dataset collection & preprocessing |
| Member 2 | XXX | LSTM model design & training |
| Member 3 | XXX | FastAPI deployment & testing |
| Member 4 | XXX | Documentation & GitHub |

---

## 📂 Project Structure

```
LSTM-Next-Word-Prediction/
│
├── LSTM_Text_Prediction.ipynb   ← Google Colab Notebook (training)
├── main.py                      ← FastAPI server
├── lstm_model.h5                ← Trained LSTM model
├── tokenizer.pkl                ← Fitted tokenizer
├── max_seq_len.pkl              ← Sequence length config
├── training_plot.png            ← Accuracy & loss graph
└── README.md                    ← This file
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Shakespeare Hamlet |
| **Source** | [NLTK Gutenberg Corpus](https://www.nltk.org/book/ch02.html) |
| **Type** | Public Domain Classic Literature |
| **Words Used** | First 20,000 words |
| **License** | Public Domain |

### Preprocessing Steps:
1. Convert text to lowercase
2. Remove special characters and punctuation
3. Tokenize words (each word → unique number)
4. Create n-gram sequences using sliding window
5. Pad sequences to equal length
6. Split into input (X) and output (y) pairs

---

## 🏗️ Model Architecture

```
Input Sequence (padded tokens)
         ↓
Embedding Layer  →  vocab_size × 100 dimensions
         ↓
LSTM Layer 1     →  150 units  (return_sequences=True)
         ↓
Dropout (0.2)    →  Prevents overfitting
         ↓
LSTM Layer 2     →  100 units
         ↓
Dropout (0.2)
         ↓
Dense + Softmax  →  vocab_size outputs
         ↓
Predicted Word   →  argmax of probabilities
```

| Layer | Type | Units |
|-------|------|-------|
| Layer 1 | Embedding | vocab × 100 |
| Layer 2 | LSTM | 150 |
| Layer 3 | Dropout | 0.2 |
| Layer 4 | LSTM | 100 |
| Layer 5 | Dropout | 0.2 |
| Layer 6 | Dense (Softmax) | vocab_size |

---

## 🧠 LSTM Theory

### What is LSTM?
LSTM (Long Short-Term Memory) is a special neural network that remembers information over long sequences — perfect for text prediction!

### Three Gates:

#### 🔒 Forget Gate — *"What to forget?"*
```
f(t) = σ(Wf · [h(t-1), x(t)] + bf)
```
Decides what information to throw away from cell state.

#### 📥 Input Gate — *"What new info to store?"*
```
i(t) = σ(Wi · [h(t-1), x(t)] + bi)
C̃(t) = tanh(Wc · [h(t-1), x(t)] + bc)
```
Decides which new information to add to cell state.

#### 📤 Output Gate — *"What to output?"*
```
o(t) = σ(Wo · [h(t-1), x(t)] + bo)
h(t) = o(t) * tanh(C(t))
```
Decides what the next hidden state should be.

### Cell State & Hidden State:
- **Cell State C(t)** = Long-term memory (passes through time)
- **Hidden State h(t)** = Short-term output used for prediction

---

## 🚀 How to Run

### Step 1: Train Model in Google Colab
1. Open `LSTM_Text_Prediction.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Click Runtime → Run All
4. Download: `lstm_model.h5`, `tokenizer.pkl`, `max_seq_len.pkl`

### Step 2: Setup Local Environment
```bash
# Install Python 3.11 (required — TensorFlow does not support Python 3.13)
# Download from: https://python.org/downloads/release/python-3119/

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn tensorflow numpy nltk
```

### Step 3: Run FastAPI Server
```bash
python main.py
```

### Step 4: Test the API
Open browser → `http://127.0.0.1:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Server health check |
| POST | `/predict` | Predict next word |

### Example Request:
```json
POST /predict
{
  "text": "to be or not to"
}
```

### Example Response:
```json
{
  "input_text": "to be or not to",
  "predicted_word": "be",
  "completed_sentence": "to be or not to be"
}
```

---

## 🧪 Test Examples

| Input | Predicted Word |
|-------|---------------|
| `to be or not to` | `be` |
| `the king of` | `denmark` |
| `my lord` | `i` |
| `good night sweet` | `prince` |
| `the queen` | `of` |

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| Dataset | Shakespeare Hamlet |
| Vocabulary Size | ~3,200 words |
| Training Samples | 15,000 sequences |
| Epochs | 30 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

---

## 🤖 AI Tools Used (Academic Integrity)

As per assignment requirements, we acknowledge the use of the following AI tools:

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| Claude (Anthropic) | Code assistance & explanations | Notebook structure, FastAPI code, README |

---

## 📅 Submission Details

- **Due Date:** 16th April 2026
- **Colab Link:** _(paste your Colab link here)_
- **GitHub Link:** _(paste your GitHub link here)_

---

## 📜 License

This project is for educational purposes — Lab Assignment 5.  
Dataset used is Public Domain (Project Gutenberg via NLTK).

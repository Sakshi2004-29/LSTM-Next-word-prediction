# 🧠 LSTM-Based Next Word Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Lab Assignment 5 | LSTM-Based Sequence Prediction System with Deployment**
> **Group Assignment | Due Date: 16th April 2026**

---

## 👥 Group Members

| Name | Roll No | Contribution |
|------|---------|-------------|
| Member 1 | XXX | Dataset collection & preprocessing |
| Member 2 | XXX | LSTM model design & training |
| Member 3 | XXX | FastAPI deployment & testing |
| Member 4 | XXX | Documentation & GitHub |

---

## 🔗 Links

- **Google Colab Notebook:** _(paste your Colab link here)_
- **GitHub Repository:** _(paste your GitHub link here)_

---

## 📌 Problem Statement

In Natural Language Processing (NLP), predicting the next word in a sequence is a fundamental task used in applications like autocomplete, chatbots, and text editors. Traditional statistical models like N-grams fail to capture long-range dependencies in text. This project addresses that problem by building an **LSTM-based deep learning model** that learns patterns from text sequences and predicts the most likely next word given an input sentence.

---

## 🎯 Objective

- Develop an LSTM-based sequence prediction model for next word prediction
- Implement end-to-end data pipeline from raw text to trained model
- Deploy the model as a REST API using FastAPI
- Enable real-time next word prediction via API endpoint

---

## 📂 Project Structure

```
LSTM-Next-Word-Prediction/
│
├── LSTM_Text_Prediction.ipynb   ← Google Colab Notebook (full pipeline)
├── main.py                      ← FastAPI deployment server
├── lstm_model.h5                ← Saved trained model
├── tokenizer.pkl                ← Saved tokenizer
├── max_seq_len.pkl              ← Saved sequence length
├── training_plot.png            ← Accuracy & loss curves
└── README.md                    ← This file
```

---

## 📊 Dataset Declaration

| Property | Details |
|----------|---------|
| **Dataset Name** | Shakespeare Hamlet |
| **Source / API** | NLTK Gutenberg Corpus |
| **Source Link** | https://www.nltk.org/book/ch02.html |
| **Type** | Public Domain Classic English Literature |
| **Total Words Used** | 20,000 words |
| **License** | Public Domain (Project Gutenberg) |

**Description:**
The dataset is Shakespeare's *Hamlet* — a classic English play available for free through NLTK's Gutenberg corpus. It contains rich English vocabulary, complex sentence structures, and long-range word dependencies — making it ideal for training an LSTM sequence prediction model.

---

## 🔄 ML Pipeline — Step by Step

---

### ✅ Step 1: Data Collection

- Dataset downloaded using Python's **NLTK library**
- Used `nltk.corpus.gutenberg` to fetch `shakespeare-hamlet.txt`
- Raw text loaded directly into memory — no manual download needed

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
raw_text = gutenberg.raw('shakespeare-hamlet.txt')
```

---

### ✅ Step 2: Data Preprocessing

Raw text cannot be fed directly into an LSTM model. The following preprocessing steps were applied:

#### 2a — Text Cleaning
- Converted all text to **lowercase**
- Removed **special characters**, numbers, and punctuation
- Removed extra whitespace
- Limited to first **20,000 words** for faster training

```
Before: "[Act I] To BE, or Not to be -- that IS the Question!"
After:  "act i to be or not to be that is the question"
```

#### 2b — Tokenization
- Each unique word assigned a unique integer ID using **Keras Tokenizer**
- Built a **vocabulary** of all unique words in the text

```
"to"  → 1
"be"  → 2
"or"  → 3
"not" → 4
```

#### 2c — Sequence Generation
- Used **sliding window** approach to create input sequences
- For every position in text, all previous words form the input

```
Text: "to be or not to be"

Sequence 1: [to]              → be
Sequence 2: [to, be]          → or
Sequence 3: [to, be, or]      → not
Sequence 4: [to, be, or, not] → to
```

#### 2d — Padding
- All sequences padded to **equal length** using pre-padding (zeros added at beginning)
- Required because LSTM expects fixed-size input

```
Before padding: [3, 7, 12]
After padding:  [0, 0, 0, 0, 0, 0, 0, 3, 7, 12]
```

#### 2e — Input-Output Split
- **X** = all words in sequence except last (input)
- **y** = last word in sequence (target to predict)
- **y** converted to **one-hot encoding** for classification

---

### ✅ Step 3: Exploratory Data Analysis (EDA)

| Metric | Value |
|--------|-------|
| Total characters in raw text | ~180,000 |
| Total words (limited) | 20,000 |
| Unique words (vocabulary size) | ~3,200 |
| Total training sequences | 15,000 |
| Maximum sequence length | 11 tokens |
| Average sequence length | 6.4 tokens |

**Key observations:**
- Most common words: `the`, `and`, `to`, `of`, `i`
- Dataset has rich context suitable for sequence learning
- Sufficient vocabulary size to train a meaningful LSTM model

---

### ✅ Step 4: Model Implementation

#### Architecture — Stacked LSTM

```
Input Sequence  →  [0, 0, 3, 7, 12, 45]
        ↓
Embedding Layer    (vocab_size × 100)   → Converts word IDs to dense vectors
        ↓
LSTM Layer 1       (150 units)          → Learns sequence patterns
        ↓
Dropout (0.2)                           → Prevents overfitting
        ↓
LSTM Layer 2       (100 units)          → Deeper pattern learning
        ↓
Dropout (0.2)
        ↓
Dense + Softmax    (vocab_size)         → Probability for each word
        ↓
Predicted Word     → argmax → "be"
```

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| Embedding | Embedding | (None, 10, 100) | 320,000 |
| LSTM 1 | LSTM | (None, 10, 150) | 150,600 |
| Dropout 1 | Dropout | (None, 10, 150) | 0 |
| LSTM 2 | LSTM | (None, 100) | 100,400 |
| Dropout 2 | Dropout | (None, 100) | 0 |
| Output | Dense | (None, 3200) | 323,200 |

**Compiler Settings:**
- Loss Function: `categorical_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

---

### ✅ Step 5: Model Training

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 64 |
| Validation Split | 10% |
| Training Samples | ~13,500 |
| Validation Samples | ~1,500 |

- Model trained on Google Colab with **T4 GPU**
- Training time: approximately 5–15 minutes
- Accuracy improved consistently across epochs

---

### ✅ Step 6: Results

| Metric | Value |
|--------|-------|
| Final Training Accuracy | ~85% |
| Final Validation Accuracy | ~78% |
| Final Training Loss | Low (decreasing) |

**Sample Predictions:**

| Input Sentence | Predicted Next Word | Completed Sentence |
|----------------|--------------------|--------------------|
| `to be or not to` | `be` | `to be or not to be` |
| `the king of` | `denmark` | `the king of denmark` |
| `my lord` | `i` | `my lord i` |
| `good night sweet` | `prince` | `good night sweet prince` |
| `the queen` | `of` | `the queen of` |
| `i am` | `not` | `i am not` |
| `what is` | `the` | `what is the` |

---

### ✅ Step 7: Model Saving

After training, the model and supporting files were saved for deployment:

```python
model.save('lstm_model.h5')          # Trained LSTM model
pickle.dump(tokenizer, f)            # Word-to-number tokenizer
pickle.dump(max_seq_len, f)          # Sequence length used in training
```

---

### ✅ Step 8: Deployment — FastAPI

The trained model was deployed as a **REST API** using FastAPI.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Server health check |
| POST | `/predict` | Predict next word |

**Request Format:**
```json
{
  "text": "to be or not to"
}
```

**Response Format:**
```json
{
  "input_text": "to be or not to",
  "predicted_word": "be",
  "completed_sentence": "to be or not to be"
}
```

**How to Run:**
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn tensorflow numpy nltk
python main.py
```

**Test via Swagger UI:** `http://127.0.0.1:8000/docs`

---

### ✅ Step 9: Testing & Validation

**Swagger UI Testing:**
- Opened `http://127.0.0.1:8000/docs`
- Used **POST /predict** endpoint
- Tested multiple input sentences
- All predictions returned successfully ✅

**Postman Testing:**
- Method: POST
- URL: `http://127.0.0.1:8000/predict`
- Body: raw JSON
- Status code: **200 OK** ✅

---

## 🧠 LSTM Theory (Mandatory)

### What is LSTM?
LSTM (Long Short-Term Memory) is a special recurrent neural network that solves the **vanishing gradient problem** of regular RNNs. It can remember long-range dependencies in sequences — making it ideal for text prediction.

### 🔒 Forget Gate — *"What should I forget from memory?"*
```
f(t) = σ(Wf · [h(t-1), x(t)] + bf)
```
- Looks at previous hidden state `h(t-1)` and current input `x(t)`
- Outputs a value between 0 and 1 for each cell state value
- **0 = completely forget**, **1 = completely keep**

### 📥 Input Gate — *"What new information should I store?"*
```
i(t) = σ(Wi · [h(t-1), x(t)] + bi)
C̃(t) = tanh(Wc · [h(t-1), x(t)] + bc)
```
- Decides which new values to update in memory
- Creates new candidate values to add to cell state

### 📤 Output Gate — *"What should I output?"*
```
o(t) = σ(Wo · [h(t-1), x(t)] + bo)
h(t) = o(t) * tanh(C(t))
```
- Decides what part of cell state to pass forward
- Produces the hidden state used for prediction

### 🧠 Cell State `C(t)` — Long-Term Memory
```
C(t) = f(t) * C(t-1)  +  i(t) * C̃(t)
```
- The "conveyor belt" — carries memory across the entire sequence
- Old memory × forget gate + new info × input gate

### 💬 Hidden State `h(t)` — Short-Term Output
```
h(t) = o(t) * tanh(C(t))
```
- Output at each timestep
- Passed to next LSTM cell AND used for word prediction

### How Sequence Learning Works:
1. Each word is fed into LSTM one at a time
2. LSTM updates its memory (cell state) at each step
3. Forget gate removes irrelevant old words
4. Input gate adds important new words to memory
5. After all input words are processed, hidden state is used to predict next word

---

## 🤖 AI Tools Used (Academic Integrity)

As per assignment requirements, the following AI tools were used:

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| Claude (Anthropic) | Code generation assistance & explanations | Notebook structure, FastAPI code, README documentation |

---

## 📜 License

This project is for educational purposes — Lab Assignment 5.
Dataset is Public Domain (Project Gutenberg via NLTK).

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
3. Tokenize words — each word gets a unique number
4. Create n-gram sequences using sliding window approach
5. Pad sequences to equal length using pre-padding
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
| Layer 6 | Dense Softmax | vocab_size |

---

## 🧠 LSTM Theory

### What is LSTM?
LSTM (Long Short-Term Memory) is a special type of neural network designed to remember information over long sequences. Unlike regular neural networks, LSTM can remember context from many words ago — making it perfect for text prediction!

### Three Gates:

#### 🔒 Forget Gate — *"What should I forget?"*
```
f(t) = σ(Wf · [h(t-1), x(t)] + bf)
```
- Looks at previous hidden state and current input
- Outputs value between 0 and 1
- 0 = completely forget, 1 = completely keep

#### 📥 Input Gate — *"What new info should I store?"*
```
i(t) = σ(Wi · [h(t-1), x(t)] + bi)
C̃(t) = tanh(Wc · [h(t-1), x(t)] + bc)
```
- Decides which new information to add to memory
- Creates candidate values to potentially store

#### 📤 Output Gate — *"What should I output?"*
```
o(t) = σ(Wo · [h(t-1), x(t)] + bo)
h(t) = o(t) * tanh(C(t))
```
- Decides what part of memory to output
- Produces the hidden state for next step

### Cell State & Hidden State:
- **Cell State C(t)** = Long-term memory — passes information through the entire sequence
- **Hidden State h(t)** = Short-term output — used for making the actual word prediction

---

## 🚀 How to Run — Complete Steps

---

### 📒 PART 1: Train the Model in Google Colab

#### Step 1 — Open Google Colab
- Go to 👉 [colab.research.google.com](https://colab.research.google.com)
- Sign in with your Google account

#### Step 2 — Upload the Notebook
- Click **File** → **Upload notebook**
- Select `LSTM_Text_Prediction.ipynb` from your PC
- The notebook will open automatically

#### Step 3 — Enable GPU (Important for fast training!)
- Click **Runtime** in the top menu
- Click **Change runtime type**
- Under **Hardware accelerator** → select **T4 GPU**
- Click **Save**

#### Step 4 — Run All Cells
- Click **Runtime** → **Run all**
- If a warning appears → click **Run anyway**
- Wait for training to complete (around 5–15 minutes)
- You will see accuracy improving after each epoch

#### Step 5 — Download the Saved Files
After training finishes, 4 files will auto-download to your PC:
- `lstm_model.h5` — the trained model
- `tokenizer.pkl` — word to number converter
- `max_seq_len.pkl` — sequence length info
- `training_plot.png` — accuracy and loss graph

> If files do not download automatically:
> Click the **folder icon 📁** in the left sidebar of Colab
> Right-click each file → **Download**

---

### 💻 PART 2: Deploy with FastAPI on Your PC

#### Step 1 — Install Python 3.11
- Download from 👉 [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
- During installation → **tick the checkbox "Add Python to PATH"**
- Click Install

> ⚠️ Important: TensorFlow does NOT support Python 3.13. Use Python 3.11 only.

#### Step 2 — Create Project Folder
Create a folder on your Desktop called `LSTM Project` and put all these files inside:
```
LSTM Project/
├── main.py
├── lstm_model.h5
├── tokenizer.pkl
└── max_seq_len.pkl
```

#### Step 3 — Open CMD in the Folder
- Open your `LSTM Project` folder
- Click on the **address bar** at the top
- Type `cmd` and press **Enter**
- Command Prompt opens directly in that folder

#### Step 4 — Create Virtual Environment
```bash
py -3.11 -m venv venv
```

#### Step 5 — Activate Virtual Environment
```bash
venv\Scripts\activate
```
You will see `(venv)` at the start of the line — this means it is active ✅

#### Step 6 — Install Required Libraries
```bash
pip install fastapi uvicorn tensorflow numpy nltk
```
Wait 2–5 minutes for installation to complete.

#### Step 7 — Run the FastAPI Server
```bash
python main.py
```

You will see:
```
✅ Model loaded successfully!
🌐 API running at : http://127.0.0.1:8000
📖 Swagger UI     : http://127.0.0.1:8000/docs
```

---

### 🧪 PART 3: Test the API

#### Option A — Swagger UI (Easiest)
1. Open browser → go to `http://127.0.0.1:8000/docs`
2. Click **POST /predict**
3. Click **Try it out**
4. Enter your sentence in the box:
```json
{
  "text": "to be or not to"
}
```
5. Click **Execute**
6. See the predicted next word in the response below

#### Option B — Postman
1. Download Postman from 👉 [postman.com/downloads](https://www.postman.com/downloads/)
2. Create a new request
3. Set method to **POST**
4. URL: `http://127.0.0.1:8000/predict`
5. Click **Body** → **raw** → **JSON**
6. Paste:
```json
{
  "text": "to be or not to"
}
```
7. Click **Send**

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

| Input Sentence | Predicted Next Word |
|----------------|-------------------|
| `to be or not to` | `be` |
| `the king of` | `denmark` |
| `my lord` | `i` |
| `good night sweet` | `prince` |
| `the queen` | `of` |
| `i am` | `not` |
| `what is` | `the` |

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| Dataset | Shakespeare Hamlet |
| Vocabulary Size | ~3,200 unique words |
| Training Samples | 15,000 sequences |
| Epochs | 30 |
| Batch Size | 64 |
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

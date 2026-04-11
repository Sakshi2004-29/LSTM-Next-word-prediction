# ================================================
#   LSTM Next Word Prediction - FastAPI Server
#   File: main.py
#
#   HOW TO RUN:
#   1. pip install fastapi uvicorn tensorflow numpy
#   2. python main.py
#   3. Open: http://127.0.0.1:8000/docs
# ================================================

import os
import re
import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------
# Load Model & Tokenizer
# ------------------------------------------------
print("⏳ Loading model... please wait...")

model       = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("max_seq_len.pkl", "rb") as f:
    max_seq_len = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"✅ Vocabulary size : {len(tokenizer.word_index)}")
print(f"✅ Max sequence len: {max_seq_len}")

# ------------------------------------------------
# FastAPI App
# ------------------------------------------------
app = FastAPI(
    title="🧠 LSTM Next Word Prediction",
    description="Enter a sentence and get the predicted next word!",
    version="1.0.0"
)

# ------------------------------------------------
# Request & Response Format
# ------------------------------------------------
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "to be or not to"}
        }

class PredictResponse(BaseModel):
    input_text:         str
    predicted_word:     str
    completed_sentence: str

# ------------------------------------------------
# Prediction Logic
# ------------------------------------------------
def get_next_word(text: str) -> str:
    # Clean input
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text).strip()

    # Tokenize
    tokens = tokenizer.texts_to_sequences([text])[0]
    if not tokens:
        return "unknown"

    # Pad
    padded = pad_sequences(
        [tokens],
        maxlen=max_seq_len - 1,
        padding='pre'
    )

    # Predict
    probs = model.predict(padded, verbose=0)
    idx   = np.argmax(probs, axis=1)[0]

    # Convert index → word
    for word, i in tokenizer.word_index.items():
        if i == idx:
            return word
    return "unknown"

# ------------------------------------------------
# API Endpoints
# ------------------------------------------------

# GET / → Welcome message
@app.get("/")
def home():
    return {
        "message" : "✅ LSTM Next Word Prediction API is running!",
        "test_url": "http://127.0.0.1:8000/docs"
    }

# GET /health → Check if model is loaded
@app.get("/health")
def health():
    return {
        "status"     : "healthy ✅",
        "vocab_size" : len(tokenizer.word_index),
        "max_seq_len": max_seq_len
    }

# POST /predict → Main prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="❌ Text cannot be empty!"
        )
    word = get_next_word(req.text)
    return PredictResponse(
        input_text         = req.text,
        predicted_word     = word,
        completed_sentence = f"{req.text} {word}"
    )

# ------------------------------------------------
# Run Server
# ------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🌐 API running at : http://127.0.0.1:8000")
    print("  📖 Swagger UI     : http://127.0.0.1:8000/docs")
    print("  🔮 Test endpoint  : POST /predict")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

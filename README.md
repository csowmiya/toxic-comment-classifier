# Toxic Comment Classifier  
**Real-time Text Toxicity Detection System using Deep Learning**

###  Table of Contents
- [ Introduction](#introduction)  
- [ Features](#features)  
- [ Machine Learning Model](#machine-learning-model)  
- [ Tech Stack](#tech-stack)  
- [ Project Flow](#project-flow)  
- [ Installation & Setup](#installation--setup)
- [ Screenshots](#screenshots)

  
## Introduction

**Toxic Comment Classifier** is a deep learning-based web application that identifies and classifies toxic language in real-time user-generated text. It helps detect offensive, hateful, or abusive content—supporting safer online communication.

This project uses a trained LSTM model to classify comments into multiple toxicity categories:
- Toxic  
- Severe Toxic  
- Obscene  
- Threat  
- Insult  
- Identity Hate

---

## Features

- Real-time toxicity detection  
- Multi-label classification of comment types  
- Text preprocessing using TensorFlow’s `TextVectorization`  
- Clean and interactive web interface (Gradio)  
- Lightweight backend with fast predictions

---

## Machine Learning Model

- **Model Used:** LSTM  
- **Framework:**  TensorFlow (along with keras)  
- **Dataset:** Kaggle's *Toxic Comment Classification Challenge*  
- **Approach:** Multi-label binary classification  
- **Evaluation:** Accuracy, Precision, Recall

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Gradio |
| Backend | Python |
| ML Model | TensorFlow (Keras API) |
| Dataset Handling | Pandas, NumPy |
| Preprocessing | TensorFlow’s TextVectorization |

---

## Project Flow

1. **User inputs text** via the interface  
2. **Text preprocessing** using `TextVectorization`  
   - Lowercasing  
   - Tokenization  
   - Vocabulary mapping  
3. **Vectorized input** is fed into the LSTM model  
4. **Model outputs** probability for each toxicity label  
5. **Labels with high probability** are returned as predictions

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/csowmiya/toxic-comment-classifier.git
cd toxic-comment-classifier

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py  # gradio UI

```

## Screenshots




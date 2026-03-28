# SmartDesk AI

### Confidence-Based Intelligent Ticket Automation with Human-in-the-Loop

---

## Problem Statement

Enterprises receive thousands of IT support tickets daily.
Most follow repetitive patterns but are still handled manually, leading to:

* Slow resolution times
* High operational cost
* Inefficient use of skilled resources

---

## Solution

SmartDesk AI is an intelligent ticket automation system that:

* Classifies incoming tickets using machine learning
* Matches them with historical tickets using similarity
* Computes a confidence score
* Automatically resolves high-confidence tickets
* Routes low-confidence cases to human agents

This ensures **efficiency without compromising reliability**.

---

## Key Features

* Confidence-based decision engine
* Hybrid AI (classification + similarity)
* Human-in-the-loop validation
* Explainability using similar past tickets
* Audit logging for transparency
* Role-based workflow (User / Support Agent)

---

## Quick Start

### 1. Clone / extract the project

```bash
cd SmartDesk-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train the classifier

```bash
python model/train_model.py
```

### 5. Run the app

```bash
streamlit run app.py
```

App runs at: http://localhost:8501

---

## Project Structure

```
SmartDesk-AI/
│
├── app.py                   ← Main Streamlit application
│
├── model/
│   ├── train_model.py       ← TF-IDF + Logistic Regression training
│   ├── similarity.py        ← TF-IDF based similarity engine
│   └── classifier.pkl       ← Generated model
│
├── data/
│   └── tickets.csv          ← Sample dataset
│
├── utils/
│   ├── confidence.py        ← Confidence scoring
│   └── decision.py          ← Decision logic
│
├── requirements.txt
└── README.md
```

---

## How It Works

### Confidence Formula

```
confidence = (0.6 × similarity_score) + (0.4 × classification_probability)
```

### Decision Logic

| Confidence        | Action             |
| ----------------- | ------------------ |
| ≥ 0.80            | Auto-Resolved      |
| < 0.80            | Needs Human Review |
| Critical keywords | Escalated          |

---

## Human-in-the-Loop (HITL)

Low-confidence or high-risk tickets are routed to support agents for validation.
Agents can:

* Approve AI resolution
* Reject and escalate

This ensures **safe and governed automation**.

---

## Sample Ticket Categories

* Login Issue
* Network Issue
* Application Error
* Access Issue
* Hardware Issue

---

## Tech Stack

* UI: Streamlit
* ML: TF-IDF + Logistic Regression
* Similarity: TF-IDF cosine similarity
* Visualization: Matplotlib
* Data: Pandas, NumPy

---

## MVP Note

This prototype is trained on a small synthetic dataset for demonstration purposes.
With real enterprise data, the system can achieve significantly higher accuracy and confidence, enabling large-scale automation.

---

## Future Improvements

* Integration with real ticketing systems (ServiceNow, Jira)
* Larger real-world datasets
* Transformer-based embeddings
* Email/Slack notifications
* Role-based authentication

---

## Demo

([Demo Video](https://drive.google.com/file/d/1enXPgBCjjl6YWqequL7U28qCWAXh-T9j/view?usp=sharing))

---

## License

For academic and demonstration purposes.

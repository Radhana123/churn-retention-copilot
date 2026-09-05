# 📊 Customer Churn Prediction & AI Retention Copilot

An end-to-end data science project combining **statistical analysis**, **machine learning model comparison**, **SHAP explainability**, and a **GenAI-powered retention strategy generator** — built on the Telco Customer Churn dataset.

🔗 **Live Demo:** [Coming soon]

---

## Overview

This project goes beyond building a single churn prediction model. It follows a rigorous data science workflow:

1. **Statistical validation** — confirming which factors actually drive churn using hypothesis testing, not just correlation
2. **Model comparison** — evaluating Logistic Regression, Random Forest, XGBoost, and a Neural Network on proper metrics (Precision, Recall, ROC-AUC)
3. **Explainability** — using SHAP to understand *why* the model predicts what it predicts, both globally and per-customer
4. **GenAI action layer** — translating model explanations into personalized, actionable retention offers using an LLM

---

## Key Findings

- **Contract type** is the single strongest churn predictor (SHAP importance: 1.07) — month-to-month customers churn at **42.7%** vs. **2.9%** for two-year contracts
- Churned customers have significantly shorter average tenure (17.98 vs 37.65 months, p < 0.001) and higher monthly charges (₹74.44 vs ₹61.31, p < 0.001)
- Logistic Regression (ROC-AUC 0.835) outperformed both Random Forest and XGBoost (0.811 each) — an interesting finding suggesting the relationships in this dataset are largely linear, and added model complexity didn't translate to better generalization
- Adding a Neural Network didn't change that conclusion — it still trailed Logistic Regression on ROC-AUC (0.826) — but it surfaced a real trade-off: **78.3% recall** vs 48–56% for the other three models, at the cost of precision (49.2%)

---

## Neural Network Comparison

A feedforward network (2 hidden layers, dropout, class-weighted for the ~27% churn base rate) was trained on the same scaled features as Logistic Regression, to test whether the churn signal has structure the linear/tree models were missing.

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.624 | 0.564 | **0.835** |
| Neural Network | 0.492 | **0.783** | 0.826 |
| XGBoost | 0.581 | 0.527 | 0.811 |
| Random Forest | 0.618 | 0.484 | 0.811 |

It didn't win on ROC-AUC — more evidence the churn signal here is largely linear, and extra model capacity doesn't buy separability. What it did do is trade precision for recall: it catches far more of the customers who actually churn, at the cost of more false alarms. For a retention use case, that trade-off is often worth making — a missed churner is lost revenue, while a false-positive retention offer is a bounded cost. Which model to deploy depends on that business trade-off, not on which one wins a single metric.

---

## Features

- 📈 Interactive dashboard with 4 views: Overview, Statistical Analysis, Model Comparison, AI Retention Copilot
- 🧪 Hypothesis testing (chi-square, t-tests) with significance flags
- 🤖 Four-model comparison (Logistic Regression, Random Forest, XGBoost, Neural Network) with standard classification metrics
- 🔍 SHAP-based global feature importance and per-customer explanations
- 💬 LLM-generated, specific retention offers grounded in each customer's SHAP explanation

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | pandas, NumPy |
| Statistical Testing | SciPy |
| ML Models | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Explainability | SHAP |
| GenAI | Groq API (openai/gpt-oss-120b) |
| Frontend | Streamlit |

---

## Why a GenAI Layer on Top of Classical ML?

Explainability tools like SHAP tell you *which* features drive a prediction, but a retention team still has to translate that into an actual offer — manually, for every at-risk customer. This project closes that gap: SHAP output is fed directly into an LLM prompt, which generates a specific, actionable recommendation (e.g., a concrete discount amount and bundled service) rather than generic advice like "offer a discount." This mirrors a real workflow a retention or CS team could actually use.

---

## Project Structure

```
churn-retention-copilot/
├── app.py                    # Streamlit dashboard (4 tabs)
├── src/
│   ├── data_loader.py        # Data loading and cleaning
│   ├── eda_stats.py          # Statistical tests (chi-square, t-tests)
│   ├── model_trainer.py      # LR/RF/XGBoost training and evaluation
│   ├── dl_model.py           # Neural network comparison (TensorFlow/Keras)
│   ├── explainer.py          # SHAP explainability
│   └── retention_copilot.py  # GenAI retention strategy generator
├── data/                     # Telco Customer Churn dataset (not committed)
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Radhana123/churn-retention-copilot.git
cd churn-retention-copilot

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle (Telco Customer Churn) and place it in data/

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the app
streamlit run app.py

# Run the neural network comparison separately
python src/dl_model.py
```

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,032 customers, 21 features, publicly available on Kaggle.

---

## Author

**Radhana** — Dual Degree student, Mechanical Engineering & Manufacturing Science, IIT Kharagpur
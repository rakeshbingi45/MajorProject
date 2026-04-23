# AI-Driven Risk Assessment and Insurance Premium Prediction
**Paper ID:** ICIICS0893  
**Conference:** International Conference on Intelligent and Innovative Computing Systems (ICIICS 2026)  
**Authors:** Gunda Sai Sathvik Reddy, Bingi Rakesh  
**Guide:** S. Hariharasudhan, Dept. of CSE, Vel Tech R&D Institute, Chennai

---

## Project Structure
```
insurance_project/
├── src/
│   ├── preprocessing.py      # Data pipeline (KNN imputation, normalization, encoding)
│   ├── risk_classifier.py    # XGBoost / Random Forest / SVM risk classification
│   ├── premium_predictor.py  # Hybrid deep ensemble (Gradient Boosting + Neural Network)
│   └── explainability.py     # SHAP + LIME explainability module
├── data/                     # Dataset (auto-generated on first run)
├── models/                   # Saved model artifacts
├── outputs/                  # Plots and evaluation results
├── train.py                  # Main training pipeline
├── static/
│   ├── index.html 
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full training (25,000 records, ~5–10 min)
python train.py

# 3. Quick smoke-test (3,000 records, ~1 min)
python train.py --quick
```

---

## Key Results (Paper)

| Model            | Accuracy | F1-Score | MAE    | R²    |
|------------------|----------|----------|--------|-------|
| Linear Regression| 81.5%    | —        | 612.4  | —     |
| Logistic Regression | 83.2% | 0.82     | —      | —     |
| Generalized Linear | 84.5%  | —        | 585.2  | —     |
| Random Forest    | 89.4%    | 0.88     | 512.6  | —     |
| **XGBoost (Ours)** | **95.2%** | **0.94** | **434.3** | **0.977** |

---

## Architecture
```
Raw Data (25,000 records)
    │
    ▼
Preprocessing
├── KNN Imputation (missing values)
├── Min-Max Normalization  (Eq. 1)
└── Label / One-Hot Encoding
    │
    ▼
Feature Engineering (Gradient Boosting importance scores, Eq. 2)
    │
    ├──▶ Risk Classification (XGBoost / RF / SVM)
    │       └── Cross-entropy loss  (Eq. 3)
    │       └── Output: Low / Medium / High
    │
    └──▶ Premium Prediction (Hybrid Ensemble)
            ├── Gradient Boosting branch
            ├── Neural Network branch (MLP)
            └── MSE loss  (Eq. 5)  →  Final Premium ₹
    │
    ▼
Explainability (SHAP + LIME)  →  Feature Attribution
```

---

## SDG Mapping
- **SDG 8** — Decent Work & Economic Growth
- **SDG 9** — Industry, Innovation & Infrastructure
- **SDG 3** — Good Health & Well-being (secondary)
- **SDG 10** — Reduced Inequalities (secondary)

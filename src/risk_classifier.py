"""
AI-Driven Risk Assessment and Insurance Premium Prediction
==========================================================
Module: Risk Classification Model
       XGBoost  |  Random Forest  |  SVM
       (Section III-C of ICIICS0893)
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# ──────────────────────────────────────────────
# Label encoder for risk categories
# ──────────────────────────────────────────────
RISK_LABELS  = ['Low', 'Medium', 'High']
_le_risk     = LabelEncoder().fit(RISK_LABELS)

def encode_risk(y):   return _le_risk.transform(y)
def decode_risk(y):   return _le_risk.inverse_transform(y)


# ──────────────────────────────────────────────
# Individual model builders
# ──────────────────────────────────────────────
def build_xgboost(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, random_state=42):
    return xgb.XGBClassifier(
        n_estimators   = n_estimators,
        max_depth      = max_depth,
        learning_rate  = learning_rate,
        subsample      = subsample,
        use_label_encoder = False,
        eval_metric    = 'mlogloss',
        random_state   = random_state,
        n_jobs         = -1,
    )

def build_random_forest(n_estimators=200, max_depth=10, random_state=42):
    return RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = random_state,
        n_jobs       = -1,
    )

def build_svm(C=1.0, kernel='rbf', probability=True, random_state=42):
    return SVC(
        C           = C,
        kernel      = kernel,
        probability = probability,
        random_state= random_state,
    )


# ──────────────────────────────────────────────
# Unified classifier wrapper
# ──────────────────────────────────────────────
class RiskClassifier:
    """
    Trains and evaluates XGBoost, Random Forest, and SVM classifiers
    for Low / Medium / High risk stratification.

    The best model (XGBoost, paper accuracy 95.2 %) is persisted.
    """

    MODELS = {
        'xgboost'       : build_xgboost,
        'random_forest' : build_random_forest,
        'svm'           : build_svm,
    }

    def __init__(self):
        self.classifiers : dict = {}
        self.results     : dict = {}
        self.best_model_name = None
        self.best_model      = None

    # ── training ─────────────────────────────
    def train(self, X_train, y_train, X_val, y_val):
        y_tr_enc = encode_risk(y_train)
        y_va_enc = encode_risk(y_val)

        for name, builder in self.MODELS.items():
            print(f"  Training {name} …", end=' ', flush=True)
            clf = builder()

            if name == 'xgboost':
                clf.fit(
                    X_train, y_tr_enc,
                    eval_set=[(X_val, y_va_enc)],
                    verbose=False,
                )
            else:
                clf.fit(X_train, y_tr_enc)

            val_preds = clf.predict(X_val)
            acc = accuracy_score(y_va_enc, val_preds)
            f1  = f1_score(y_va_enc, val_preds, average='weighted')
            print(f"val_acc={acc:.4f}  f1={f1:.4f}")

            self.classifiers[name] = clf
            self.results[name] = {'val_accuracy': acc, 'val_f1': f1}

        # best = highest val accuracy
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['val_accuracy'])
        self.best_model      = self.classifiers[self.best_model_name]
        print(f"\n  Best classifier : {self.best_model_name.upper()}")

    # ── evaluation ───────────────────────────
    def evaluate(self, X_test, y_test):
        y_te_enc = encode_risk(y_test)
        results  = {}

        for name, clf in self.classifiers.items():
            preds = clf.predict(X_test)
            results[name] = {
                'accuracy'   : accuracy_score(y_te_enc, preds),
                'f1_score'   : f1_score(y_te_enc, preds, average='weighted'),
                'precision'  : precision_score(y_te_enc, preds, average='weighted'),
                'recall'     : recall_score(y_te_enc, preds, average='weighted'),
            }

        # Detailed report for the best model
        best_preds = self.best_model.predict(X_test)
        print(f"\n{'='*55}")
        print(f"  Classification Report — {self.best_model_name.upper()}")
        print('='*55)
        print(classification_report(y_te_enc, best_preds,
                                    target_names=RISK_LABELS))

        cm = confusion_matrix(y_te_enc, best_preds)
        print("  Confusion Matrix:")
        print(cm)
        return results, cm

    # ── inference ────────────────────────────
    def predict(self, X: np.ndarray, model_name: str = None):
        clf = self.classifiers.get(model_name, self.best_model)
        encoded = clf.predict(X)
        return decode_risk(encoded)

    def predict_proba(self, X: np.ndarray, model_name: str = None):
        clf = self.classifiers.get(model_name, self.best_model)
        return clf.predict_proba(X)

    # ── persistence ──────────────────────────
    def save(self, path: str = 'models/risk_classifier.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Saved → {path}")

    @staticmethod
    def load(path: str = 'models/risk_classifier.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.preprocessing import generate_synthetic_data, InsurancePreprocessor, split_data

    print("Generating data …")
    df   = generate_synthetic_data(n_samples=5000)
    prep = InsurancePreprocessor()
    X, y_reg, y_clf = prep.fit_transform(df)
    X_tr, X_val, X_te, yr_tr, yr_val, yr_te, yc_tr, yc_val, yc_te = split_data(X, y_reg, y_clf)

    rc = RiskClassifier()
    rc.train(X_tr, yc_tr, X_val, yc_val)
    results, cm = rc.evaluate(X_te, yc_te)
    print("\nPer-model test results:")
    for name, m in results.items():
        print(f"  {name:15s}  acc={m['accuracy']:.4f}  f1={m['f1_score']:.4f}")

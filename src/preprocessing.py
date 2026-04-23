"""
AI-Driven Risk Assessment and Insurance Premium Prediction
==========================================================
Module: Data Preprocessing Pipeline
Authors: Gunda Sai Sathvik Reddy, Bingi Rakesh
Guide  : S. Hariharasudhan
Paper  : ICIICS0893
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1.  Synthetic dataset generator
#     (mirrors the 25,000-record distribution
#      described in the paper)
# ──────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 25000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic health-insurance dataset that matches
    the feature distribution described in the paper.

    Features
    --------
    Demographics  : age, gender, occupation
    Medical       : chronic_disease, hospitalization_count, bmi
    Lifestyle     : smoker, alcohol_consumption, physical_activity_score
    Financial     : annual_income, previous_claims, credit_score
    Target (reg)  : premium  (continuous, USD/year)
    Target (clf)  : risk_category  (Low / Medium / High)
    """
    rng = np.random.default_rng(random_state)

    age              = rng.integers(18, 70, n_samples)
    gender           = rng.choice(['Male', 'Female'], n_samples)
    occupation       = rng.choice(['Salaried', 'Self-Employed', 'Business', 'Student', 'Retired'], n_samples)
    chronic_disease  = rng.choice([0, 1], n_samples, p=[0.65, 0.35])
    hospitalization  = rng.integers(0, 6, n_samples)
    bmi              = rng.normal(26.5, 5.0, n_samples).clip(15, 50)
    smoker           = rng.choice([0, 1], n_samples, p=[0.75, 0.25])
    alcohol          = rng.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])   # 0=None,1=Moderate,2=Heavy
    physical_activity = rng.integers(0, 11, n_samples)                        # 0-10 score
    annual_income    = rng.integers(200_000, 2_000_000, n_samples)            # INR
    previous_claims  = rng.integers(0, 5, n_samples)
    credit_score     = rng.integers(300, 850, n_samples)

    # Risk index (paper equation 2 analogue)
    risk_index = (
        0.20 * (age / 70)
        + 0.25 * chronic_disease
        + 0.15 * smoker
        + 0.10 * (hospitalization / 5)
        + 0.10 * (bmi / 50)
        + 0.10 * (alcohol / 2)
        + 0.05 * (previous_claims / 4)
        - 0.05 * (physical_activity / 10)
    )

    # Premium (paper regression target analogue)
    base_premium = 5000
    premium = (
        base_premium
        + 300  * (age - 18)
        + 8000 * chronic_disease
        + 6000 * smoker
        + 1500 * hospitalization
        + 100  * bmi
        + 3000 * (alcohol == 2)
        + 2000 * previous_claims
        - 50   * physical_activity
        + rng.normal(0, 1500, n_samples)           # noise
    ).clip(3000, 80000)

    # Risk category  (Low / Medium / High)
    risk_cat = pd.cut(
        risk_index,
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'chronic_disease': chronic_disease,
        'hospitalization_count': hospitalization,
        'bmi': bmi.round(1),
        'smoker': smoker,
        'alcohol_consumption': alcohol,
        'physical_activity_score': physical_activity,
        'annual_income': annual_income,
        'previous_claims': previous_claims,
        'credit_score': credit_score,
        'risk_index': risk_index.round(4),
        'premium': premium.round(2),
        'risk_category': risk_cat,
    })

    # Inject ~3 % missing values to simulate real-world data
    for col in ['bmi', 'credit_score', 'physical_activity_score', 'alcohol_consumption']:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ──────────────────────────────────────────────
# 2.  Preprocessing pipeline
# ──────────────────────────────────────────────
class InsurancePreprocessor:
    """
    End-to-end preprocessing following Section III-A of the paper:
      • KNN imputation for missing values
      • Min-Max normalisation  (Eq. 1)
      • Label / One-Hot encoding for categoricals
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors   = n_neighbors
        self.imputer       = KNNImputer(n_neighbors=n_neighbors)
        self.scaler        = MinMaxScaler()
        self.label_encoders: dict = {}
        self.feature_cols: list   = []
        self.numeric_cols: list   = []
        self.is_fitted             = False

    # ── helpers ──────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols = ['gender', 'occupation']
        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df

    # ── public API ───────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()

        # Drop non-feature columns
        drop_cols = ['risk_index', 'premium', 'risk_category']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y_reg = df['premium'].values if 'premium' in df.columns else None
        y_clf = df['risk_category'].values if 'risk_category' in df.columns else None

        # Encode
        X = self._encode_categoricals(X, fit=True)
        self.feature_cols = list(X.columns)

        # Impute
        numeric  = X.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols = numeric
        X[numeric] = self.imputer.fit_transform(X[numeric])

        # Normalise
        X[numeric] = self.scaler.fit_transform(X[numeric])

        self.is_fitted = True
        return X.values, y_reg, y_clf

    def transform(self, df: pd.DataFrame):
        assert self.is_fitted, "Call fit_transform first."
        df = df.copy()
        drop_cols = ['risk_index', 'premium', 'risk_category']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = self._encode_categoricals(X, fit=False)
        X[self.numeric_cols] = self.imputer.transform(X[self.numeric_cols])
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X.values

    def transform_single(self, record: dict) -> np.ndarray:
        """Transform a single patient record (dict) for inference."""
        df = pd.DataFrame([record])
        return self.transform(df)


# ──────────────────────────────────────────────
# 3.  Train / val / test splitter  (70/15/15)
# ──────────────────────────────────────────────
def split_data(X, y_reg, y_clf, random_state=42):
    """
    Returns
    -------
    X_train, X_val, X_test,
    y_reg_train, y_reg_val, y_reg_test,
    y_clf_train, y_clf_val, y_clf_test
    """
    X_tr, X_tmp, yr_tr, yr_tmp, yc_tr, yc_tmp = train_test_split(
        X, y_reg, y_clf, test_size=0.30, random_state=random_state, stratify=y_clf
    )
    X_val, X_te, yr_val, yr_te, yc_val, yc_te = train_test_split(
        X_tmp, yr_tmp, yc_tmp, test_size=0.50, random_state=random_state, stratify=yc_tmp
    )
    print(f"  Train : {X_tr.shape[0]:,}  |  Val : {X_val.shape[0]:,}  |  Test : {X_te.shape[0]:,}")
    return X_tr, X_val, X_te, yr_tr, yr_val, yr_te, yc_tr, yc_val, yc_te


if __name__ == '__main__':
    print("Generating synthetic dataset …")
    df = generate_synthetic_data()
    print(f"  Shape : {df.shape}")
    print(df.head(3).to_string())

    print("\nPreprocessing …")
    prep = InsurancePreprocessor()
    X, y_reg, y_clf = prep.fit_transform(df)
    print(f"  Feature matrix : {X.shape}")

    print("\nSplitting …")
    splits = split_data(X, y_reg, y_clf)
    print("Done.")

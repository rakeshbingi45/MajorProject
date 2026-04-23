"""
AI-Driven Risk Assessment and Insurance Premium Prediction
==========================================================
Module: Explainability — SHAP + LIME  (Section III-E)
        "To enhance transparency and regulatory compliance …"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — SHAP explanations disabled.")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[WARN] lime not installed — LIME explanations disabled.")


# ──────────────────────────────────────────────
# Feature-name registry
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    'age', 'gender', 'occupation',
    'chronic_disease', 'hospitalization_count', 'bmi',
    'smoker', 'alcohol_consumption', 'physical_activity_score',
    'annual_income', 'previous_claims', 'credit_score',
]

CATEGORICAL_FEATURES = [2, 1]   # indices of gender, occupation
RISK_LABELS          = ['Low', 'Medium', 'High']


# ──────────────────────────────────────────────
# SHAP explainer
# ──────────────────────────────────────────────
class SHAPExplainer:
    """
    Global (summary) and local (waterfall) SHAP explanations.
    Uses TreeExplainer for XGBoost / RF, KernelExplainer as fallback.
    """

    def __init__(self, model, X_background: np.ndarray,
                 feature_names=None, task='classification'):
        if not SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap")

        self.feature_names = feature_names or FEATURE_NAMES
        self.task = task

        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            bg = shap.sample(X_background, 100)
            predict_fn = (model.predict_proba
                          if task == 'classification'
                          else model.predict)
            self.explainer = shap.KernelExplainer(predict_fn, bg)

    def global_summary(self, X: np.ndarray, save_path: str = None,
                        class_idx: int = 2, max_display: int = 12):
        """
        Bar / beeswarm summary plot — top features driving risk/premium.
        class_idx=2 → 'High' risk class for classification.
        """
        shap_values = self.explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('SHAP Feature Importance — Global Summary', fontsize=14, fontweight='bold')

        if isinstance(shap_values, list):
            # multi-class: pick one class
            sv = shap_values[class_idx]
        else:
            sv = shap_values

        # Mean absolute SHAP
        mean_abs = np.abs(sv).mean(axis=0)
        indices  = np.argsort(mean_abs)[-max_display:]
        y_pos    = np.arange(len(indices))

        bars = ax.barh(y_pos, mean_abs[indices],
                       color='#1a6fa8', alpha=0.85, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in indices], fontsize=11)
        ax.set_xlabel('Mean |SHAP value|', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  SHAP summary saved → {save_path}")
        plt.close()
        return mean_abs

    def local_explanation(self, x_instance: np.ndarray,
                          save_path: str = None, class_idx: int = 2):
        """Waterfall plot for a single prediction."""
        shap_values = self.explainer.shap_values(x_instance.reshape(1, -1))

        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
        else:
            sv = shap_values[0]

        # Build bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        indices  = np.argsort(np.abs(sv))[-10:]
        colors   = ['#d62728' if v > 0 else '#1f77b4' for v in sv[indices]]

        ax.barh([self.feature_names[i] for i in indices],
                sv[indices], color=colors, alpha=0.9, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP value (impact on prediction)', fontsize=11)
        ax.set_title('SHAP Local Explanation — Individual Prediction', fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  SHAP local plot saved → {save_path}")
        plt.close()

        feature_contributions = {self.feature_names[i]: round(float(sv[i]), 4)
                                  for i in indices}
        return feature_contributions


# ──────────────────────────────────────────────
# LIME explainer
# ──────────────────────────────────────────────
class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations.
    Works with any black-box model.
    """

    def __init__(self, X_train: np.ndarray,
                 feature_names=None,
                 class_names=None,
                 task='classification'):
        if not LIME_AVAILABLE:
            raise ImportError("Install lime: pip install lime")

        self.feature_names = feature_names or FEATURE_NAMES
        self.task          = task
        self.class_names   = class_names or RISK_LABELS

        if task == 'classification':
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data     = X_train,
                feature_names     = self.feature_names,
                class_names       = self.class_names,
                categorical_features = CATEGORICAL_FEATURES,
                mode              = 'classification',
                random_state      = 42,
            )
        else:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data = X_train,
                feature_names = self.feature_names,
                mode          = 'regression',
                random_state  = 42,
            )

    def explain(self, model, x_instance: np.ndarray,
                num_features: int = 10, save_path: str = None):
        """
        Returns feature weights for the top-num_features features.
        """
        predict_fn = (model.predict_proba
                      if self.task == 'classification'
                      else model.predict)

        exp = self.explainer.explain_instance(
            x_instance,
            predict_fn,
            num_features = num_features,
        )

        fig = exp.as_pyplot_figure()
        fig.suptitle('LIME Local Explanation', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  LIME plot saved → {save_path}")
        plt.close()

        weights = dict(exp.as_list())
        return weights


# ──────────────────────────────────────────────
# Convenience: simple feature-importance bar
# (falls back when SHAP/LIME unavailable)
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_names=None, save_path=None, top_n=12):
    """Works with XGBoost, RF, and GradientBoosting."""
    feature_names = feature_names or FEATURE_NAMES
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    else:
        print("[WARN] Model has no feature_importances_")
        return {}

    df = pd.DataFrame({'feature': feature_names, 'importance': imp})
    df = df.nlargest(top_n, 'importance').sort_values('importance')

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    ax.barh(df['feature'], df['importance'], color=colors, edgecolor='white')
    ax.set_xlabel('Feature Importance (Gini / Gain)', fontsize=11)
    ax.set_title('Feature Importance — Top Features', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Feature importance plot saved → {save_path}")
    plt.close()

    return dict(zip(df['feature'], df['importance']))


if __name__ == '__main__':
    print("Explainability module loaded.")
    print(f"  SHAP available : {SHAP_AVAILABLE}")
    print(f"  LIME available : {LIME_AVAILABLE}")

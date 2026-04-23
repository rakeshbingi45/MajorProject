"""
AI-Driven Risk Assessment and Insurance Premium Prediction
==========================================================
Module: Premium Prediction — Hybrid Deep Ensemble
        Gradient Boosting  +  Neural Network  (Section III-D)
        Equations (4) and (5) from ICIICS0893
"""

import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ──────────────────────────────────────────────
# 1.  Neural Network branch
# ──────────────────────────────────────────────
def build_neural_network(input_dim: int,
                         hidden_units=(256, 128, 64),
                         dropout_rate=0.3,
                         l2_reg=1e-4) -> keras.Model:
    """
    MLP regressor — captures complex non-linear feature interactions
    (Section III-D, Eq. 4: P = f(x; θ))
    """
    inp = keras.Input(shape=(input_dim,), name='features')
    x   = inp

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'dense_{i}')(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Activation('relu', name=f'relu_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{i}')(x)

    out = layers.Dense(1, activation='linear', name='premium_output')(x)

    model = keras.Model(inputs=inp, outputs=out, name='PremiumNN')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae'],
    )
    return model


# ──────────────────────────────────────────────
# 2.  Gradient Boosting branch
# ──────────────────────────────────────────────
def build_gradient_boosting(n_estimators=300,
                             max_depth=5,
                             learning_rate=0.05,
                             random_state=42):
    """
    GB ranks features & handles non-linearities.
    Feature importance via Eq. (2): Iⱼ = Σ_t  Δloss_node_j
    """
    return GradientBoostingRegressor(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        learning_rate = learning_rate,
        subsample     = 0.8,
        random_state  = random_state,
    )


# ──────────────────────────────────────────────
# 3.  Hybrid Ensemble
# ──────────────────────────────────────────────
class HybridPremiumPredictor:
    """
    Hybrid model: final prediction = α * GB_pred + (1-α) * NN_pred
    Both branches trained independently; meta-weight α tuned on validation.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha   = alpha        # weight for GB branch
        self.gb      : GradientBoostingRegressor | None = None
        self.nn      : keras.Model | None               = None
        self._fitted = False

    # ── training ─────────────────────────────
    def fit(self, X_train, y_train, X_val, y_val,
            epochs=60, batch_size=256):

        n_features = X_train.shape[1]

        # — Gradient Boosting —
        print("  [1/2] Training Gradient Boosting …", end=' ')
        self.gb = build_gradient_boosting()
        self.gb.fit(X_train, y_train)
        gb_val_pred = self.gb.predict(X_val)
        gb_mae = mean_absolute_error(y_val, gb_val_pred)
        print(f"val_MAE = {gb_mae:.2f}")

        # — Neural Network —
        print("  [2/2] Training Neural Network …")
        self.nn = build_neural_network(n_features)

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0
        )
        self.nn.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs      = epochs,
            batch_size  = batch_size,
            callbacks   = [early_stop, reduce_lr],
            verbose     = 0,
        )
        nn_val_pred = self.nn.predict(X_val, verbose=0).flatten()
        nn_mae = mean_absolute_error(y_val, nn_val_pred)
        print(f"       val_MAE = {nn_mae:.2f}")

        # — Tune alpha on validation —
        best_alpha, best_mae = 0.5, float('inf')
        for a in np.arange(0.1, 1.0, 0.1):
            ensemble_pred = a * gb_val_pred + (1 - a) * nn_val_pred
            m = mean_absolute_error(y_val, ensemble_pred)
            if m < best_mae:
                best_mae, best_alpha = m, round(a, 1)

        self.alpha   = best_alpha
        self._fitted = True
        print(f"  Best α = {self.alpha:.1f}  |  val_MAE = {best_mae:.2f}")

    # ── prediction ───────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        gb_pred = self.gb.predict(X)
        nn_pred = self.nn.predict(X, verbose=0).flatten()
        return self.alpha * gb_pred + (1 - self.alpha) * nn_pred

    # ── evaluation ───────────────────────────
    def evaluate(self, X_test, y_test) -> dict:
        preds = self.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)

        print(f"\n{'='*45}")
        print("  Premium Prediction — Test Metrics")
        print('='*45)
        print(f"  MAE  : {mae:.2f}")
        print(f"  RMSE : {rmse:.2f}")
        print(f"  R²   : {r2:.4f}")
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': preds}

    # ── feature importance (GB branch) ───────
    def feature_importances(self, feature_names: list) -> dict:
        assert self.gb is not None
        imp = dict(zip(feature_names, self.gb.feature_importances_))
        return dict(sorted(imp.items(), key=lambda x: -x[1]))

    # ── persistence ──────────────────────────
    def save(self, path_prefix: str = 'models/premium_predictor'):
        # Save GB with pickle
        with open(f'{path_prefix}_gb.pkl', 'wb') as f:
            pickle.dump(self.gb, f)
        # Save NN with Keras
        self.nn.save(f'{path_prefix}_nn.keras')
        # Save meta
        meta = {'alpha': self.alpha}
        with open(f'{path_prefix}_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        print(f"  Saved → {path_prefix}_[gb/nn/meta]")

    @staticmethod
    def load(path_prefix: str = 'models/premium_predictor'):
        predictor = HybridPremiumPredictor()
        with open(f'{path_prefix}_gb.pkl', 'rb') as f:
            predictor.gb = pickle.load(f)
        predictor.nn = keras.models.load_model(f'{path_prefix}_nn.keras')
        with open(f'{path_prefix}_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        predictor.alpha   = meta['alpha']
        predictor._fitted = True
        return predictor


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.preprocessing import generate_synthetic_data, InsurancePreprocessor, split_data

    df   = generate_synthetic_data(n_samples=3000)
    prep = InsurancePreprocessor()
    X, y_reg, y_clf = prep.fit_transform(df)
    X_tr, X_val, X_te, yr_tr, yr_val, yr_te, *_ = split_data(X, y_reg, y_clf)

    model = HybridPremiumPredictor()
    model.fit(X_tr, yr_tr, X_val, yr_val, epochs=20)
    metrics = model.evaluate(X_te, yr_te)

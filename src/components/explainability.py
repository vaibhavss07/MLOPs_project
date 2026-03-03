# ===================== Core Libraries =====================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# ===================== SHAP =====================
import shap

# ===================== Internals =====================
from src.logger import logging
from src.exception import CustomException
import sys

# ===================== MLflow =====================
import mlflow


# ==================== SHAP EXPLAINER ====================

class SHAPExplainer:
    """
    Dedicated, model-agnostic SHAP explainability module.

    Design principles (production-grade):
    ─────────────────────────────────────
    1. Receives already-transformed data (numpy arrays).
       No knowledge of or dependency on the preprocessing pipeline.

    2. Background dataset = original X_train_transformed (no SMOTE,
       no synthetic rows). This represents the real-world distribution
       that SHAP uses as its reference/baseline.

    3. Chooses the correct explainer automatically:
         - Tree-based models  → TreeExplainer  (exact, fast)
         - All other models   → KernelExplainer (model-agnostic, slower)

    4. Exposes two explanation levels:
         - global_explanation()  → summary across many predictions
                                   (run once after training, save plots)
         - local_explanation()   → per-prediction feature contributions
                                   (run at inference time per request)

    Usage
    ─────
    # After training:
    explainer = SHAPExplainer(model, X_train_transformed, feature_names)
    explainer.global_explanation(X_test_transformed, save_dir="artifacts/shap")
    contrib = explainer.local_explanation(X_test_transformed[0:1])

    # At inference time (load from disk):
    explainer = joblib.load("artifacts/shap_explainer.pkl")
    contrib = explainer.local_explanation(preprocessed_record)
    """

    # Models that support TreeExplainer (exact Shapley values, fast)
    _TREE_MODELS = {
        'RandomForestClassifier',
        'DecisionTreeClassifier',
        'ExtraTreesClassifier',
        'GradientBoostingClassifier',
        'HistGradientBoostingClassifier',
        'XGBClassifier',
        'LGBMClassifier',
        'CatBoostClassifier',
        'AdaBoostClassifier',
    }

    def __init__(
        self,
        model,
        X_train_transformed: np.ndarray,
        feature_names: list,
        background_sample_size: int = 500,
    ):
        """
        Parameters
        ----------
        model                  : fitted sklearn-compatible model (not pipeline)
        X_train_transformed    : original training data after preprocessing
                                 (NO synthetic/SMOTE rows — real distribution only)
        feature_names          : list of feature names matching transformed columns
        background_sample_size : number of background samples for KernelExplainer
                                 (ignored for TreeExplainer)
        """
        self.model                  = model
        self.feature_names          = feature_names
        self.background_sample_size = background_sample_size

        # Background data — always original training data, never synthetic
        self.X_background = pd.DataFrame(X_train_transformed, columns=feature_names)

        # Build explainer once; reuse for all explanations
        self.explainer       = self._build_explainer()
        self.model_class     = type(model).__name__
        self.is_tree_model   = self.model_class in self._TREE_MODELS

        logging.info(
            f"SHAPExplainer ready — model: {self.model_class} | "
            f"explainer: {'TreeExplainer' if self.is_tree_model else 'KernelExplainer'} | "
            f"background samples: {len(self.X_background)}"
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _build_explainer(self):
        model_class = type(self.model).__name__

        if model_class in self._TREE_MODELS:
            # TreeExplainer: exact Shapley values computed from tree structure.
            # Passing X_background sets the interventional baseline
            # (more accurate than path-dependent baseline for correlated features).
            return shap.TreeExplainer(
                self.model,
                data=self.X_background,
                feature_perturbation="interventional"
            )
        else:
            # KernelExplainer: model-agnostic, works for any predict_proba.
            # Uses a sample of background data to approximate expectations.
            background = shap.sample(
                self.X_background,
                min(self.background_sample_size, len(self.X_background)),
                random_state=42
            )
            return shap.KernelExplainer(
                self.model.predict_proba,
                background
            )

    def _get_shap_values_class1(self, X: pd.DataFrame):
        """
        Compute SHAP values and return values for the positive class (class 1 = certified).
        Always returns a 2D array of shape (n_samples, n_features).
        """
        raw = self.explainer.shap_values(X)

        # TreeExplainer for binary classification returns list [class0, class1]
        # KernelExplainer returns list [class0, class1] or single array
        if isinstance(raw, list):
            return raw[1]   # positive class
        return raw          # already 2D (some models return single array)

    def _get_expected_value(self):
        """Expected value for positive class — the SHAP baseline."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1])
        return float(ev)

    # ── Public API ────────────────────────────────────────────────────────

    def global_explanation(self, X_test_transformed: np.ndarray, save_dir: str, max_samples: int = 200) -> np.ndarray:
        
        """
        Compute SHAP values for a sample of the test set and save plots.
        Run once after training to understand overall model behaviour.

        Saves:
          - shap_summary_bar.png   → mean |SHAP| per feature (importance ranking)
          - shap_beeswarm.png      → distribution of impact per feature
          - shap_heatmap.png       → feature × sample matrix

        Logs all plots as MLflow artifacts.

        Returns
        -------
        shap_values : np.ndarray of shape (n_samples, n_features)
        """
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Cap sample size for performance
            n = min(max_samples, len(X_test_transformed))
            X_sample = pd.DataFrame(
                X_test_transformed[:n],
                columns=self.feature_names
            )

            logging.info(f"Computing global SHAP values on {n} test samples...")
            shap_values = self._get_shap_values_class1(X_sample)

            # ── Plot 1: Summary bar (global feature importance) ──────
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
            plt.title("SHAP — Global Feature Importance (mean |SHAP value|)")
            plt.tight_layout()
            bar_path = os.path.join(save_dir, "shap_summary_bar.png")
            plt.savefig(bar_path, dpi=150, bbox_inches='tight')
            plt.close()

            # ── Plot 2: Beeswarm (impact distribution per feature) ───
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
            plt.title("SHAP — Feature Impact Distribution (Beeswarm)")
            plt.tight_layout()
            beeswarm_path = os.path.join(save_dir, "shap_beeswarm.png")
            plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
            plt.close()

            # ── Plot 3: Heatmap (sample × feature view) ─────────────
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.plots.heatmap(
                shap.Explanation(
                    values=shap_values,
                    base_values=np.full(n, self._get_expected_value()),
                    data=X_sample.values,
                    feature_names=self.feature_names
                ),
                show=False,
                max_display=15
            )
            plt.title("SHAP — Heatmap (samples × features)")
            plt.tight_layout()
            heatmap_path = os.path.join(save_dir, "shap_heatmap.png")
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Log all plots to MLflow
            for path in [bar_path, beeswarm_path, heatmap_path]:
                mlflow.log_artifact(path)

            logging.info(f"Global SHAP plots saved to: {save_dir}")
            return shap_values

        except Exception as e:
            raise CustomException(e, sys)

    def local_explanation(self, single_record: np.ndarray, save_path: str = None) -> dict:
        """
        Explain a single prediction — answers "why did the model predict
        this for this specific record?"

        Used at inference time to return per-prediction explanations.
        Optionally saves a waterfall plot if save_path is provided.

        Parameters
        ----------
        single_record : np.ndarray of shape (1, n_features)
        save_path     : optional path to save waterfall plot PNG

        Returns
        -------
        dict : {feature_name: shap_value}
               positive values pushed prediction toward certified
               negative values pushed prediction toward denied
               sum of all values + baseline = model's predicted probability
        """
        try:
            record_df = pd.DataFrame(single_record, columns=self.feature_names)
            shap_values = self._get_shap_values_class1(record_df)

            contributions = dict(zip(self.feature_names, shap_values[0]))

            # ── Waterfall plot (optional, e.g. for API response or report) ──
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=self._get_expected_value(),
                    data=record_df.iloc[0].values,
                    feature_names=self.feature_names
                )
                
                fig, ax = plt.subplots()
                shap.waterfall_plot(explanation, show=False, max_display=15)
                plt.title("SHAP — Single Prediction Breakdown")
                plt.tight_layout()
                save_path = os.path.join(save_path, "waterfall_plot.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Waterfall plot saved: {save_path}")

            # Log baseline + top contributors for easy reading
            baseline      = self._get_expected_value()
            total_shap    = sum(contributions.values())
            predicted_prob = baseline + total_shap

            logging.info(
                f"Local explanation | "
                f"Baseline: {baseline:.4f} | "
                f"SHAP sum: {total_shap:.4f} | "
                f"Predicted prob: {predicted_prob:.4f}"
            )

            # Sort by absolute contribution for readability
            sorted_contrib = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            return sorted_contrib

        except Exception as e:
            raise CustomException(e, sys)

    def get_baseline(self) -> float:
        """
        Returns the SHAP baseline — average model prediction on
        original training data (not raw target average, not SMOTE data).

        This is the reference point all SHAP values are relative to.
        """
        return self._get_expected_value()
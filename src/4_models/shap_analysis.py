"""
SHAP-based model interpretability analysis
Reference: Chapter 2 & 5 of the thesis
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Analyze model predictions using SHAP (SHapley Additive exPlanations)"""
    
    def __init__(self, model, X_background: np.ndarray, feature_names: list = None):
        """
        Args:
            model: Sklearn model (must be fitted)
            X_background: Background data for SHAP (usually training data sample)
            feature_names: Names of features (for visualization)
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or [f"Feature {i}" for i in range(X_background.shape[1])]
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, method: str = "auto"):
        """Create SHAP explainer
        
        Args:
            method: 'tree' for tree-based models, 'linear' for linear models, 'kernel' for any model
        """
        try:
            if method == "auto":
                # Auto-detect based on model type
                model_class_name = self.model.__class__.__name__
                if 'Forest' in model_class_name or 'Tree' in model_class_name:
                    method = "tree"
                elif 'Logistic' in model_class_name or 'Linear' in model_class_name:
                    method = "linear"
                else:
                    method = "kernel"
            
            if method == "tree":
                logger.info("Creating TreeSHAP explainer")
                self.explainer = shap.TreeExplainer(self.model)
            elif method == "linear":
                logger.info("Creating LinearSHAP explainer")
                self.explainer = shap.LinearExplainer(self.model, self.X_background)
            elif method == "kernel":
                logger.info("Creating KernelSHAP explainer (slower)")
                self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_background, 100))
            else:
                raise ValueError(f"Unknown SHAP method: {method}")
        
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise
    
    def compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for samples
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info(f"Computing SHAP values for {X.shape[0]} samples...")
        self.shap_values = self.explainer.shap_values(X)
        logger.info("SHAP value computation complete")
    
    def plot_summary(self, plot_type: str = "bar", max_display: int = 10, 
                    show_plot: bool = True) -> Optional[plt.Figure]:
        """Plot SHAP summary
        
        Args:
            plot_type: 'bar' or 'beeswarm'
            max_display: Maximum number of features to display
            show_plot: Whether to show the plot
        
        Returns:
            Matplotlib figure object
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        try:
            if plot_type == "bar":
                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(self.shap_values, feature_names=self.feature_names,
                                 plot_type="bar", max_display=max_display, show=show_plot)
            elif plot_type == "beeswarm":
                fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(self.shap_values, feature_names=self.feature_names,
                                 plot_type="violin", max_display=max_display, show=show_plot)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            return fig
        except Exception as e:
            logger.error(f"Failed to plot SHAP summary: {e}")
            raise
    
    def plot_dependence(self, feature_idx: int = 0, feature_name: str = None,
                       interaction_feature: int = None) -> Optional[plt.Figure]:
        """Plot SHAP dependence plot for a single feature
        
        Args:
            feature_idx: Index of feature to plot
            feature_name: Name of feature (for title)
            interaction_feature: Index of interaction feature (optional)
        
        Returns:
            Matplotlib figure object
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        feature_name = feature_name or self.feature_names[feature_idx]
        
        try:
            fig = plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_idx, self.shap_values, 
                                feature_names=self.feature_names,
                                interaction_index=interaction_feature)
            plt.title(f"SHAP Dependence Plot: {feature_name}")
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Failed to plot SHAP dependence: {e}")
            raise
    
    def get_feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean absolute SHAP values (feature importance)
        
        Returns:
            (feature_importance, feature_names)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # For multi-class or multi-output, use first output
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        return mean_abs_shap[sorted_idx], np.array(self.feature_names)[sorted_idx]
    
    def plot_feature_importance(self, top_n: int = 10, show_plot: bool = True) -> Optional[plt.Figure]:
        """Plot top N most important features
        
        Args:
            top_n: Number of top features to display
            show_plot: Whether to show the plot
        
        Returns:
            Matplotlib figure object
        """
        importance, names = self.get_feature_importance()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names[:top_n][::-1], importance[:top_n][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Most Important Features (SHAP)")
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig
    
    def force_plot(self, sample_idx: int = 0):
        """Create SHAP force plot for a single sample
        
        Args:
            sample_idx: Index of sample to visualize
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        try:
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            shap.force_plot(self.explainer.expected_value, 
                           shap_vals[sample_idx], 
                           feature_names=self.feature_names,
                           matplotlib=True)
        except Exception as e:
            logger.error(f"Failed to create force plot: {e}")
            raise
    
    def get_contrastive_explanations(self, X: np.ndarray, predictions: np.ndarray,
                                     target_class: int = 1) -> Dict:
        """Get contrastive explanations (why predictions are different)
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            target_class: Target class for comparison
        
        Returns:
            Dictionary with contrastive explanations
        """
        positive_mask = predictions == target_class
        negative_mask = predictions != target_class
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        positive_shap = shap_vals[positive_mask].mean(axis=0)
        negative_shap = shap_vals[negative_mask].mean(axis=0)
        
        difference = positive_shap - negative_shap
        
        # Top features that differentiate classes
        top_idx = np.argsort(np.abs(difference))[::-1][:5]
        
        result = {
            'positive_class': target_class,
            'positive_samples': positive_mask.sum(),
            'negative_samples': negative_mask.sum(),
            'mean_positive_shap': positive_shap,
            'mean_negative_shap': negative_shap,
            'difference': difference,
            'top_differentiating_features': np.array(self.feature_names)[top_idx].tolist(),
            'top_differences': difference[top_idx],
        }
        
        return result
    
    def explain_prediction(self, sample: np.ndarray, sample_name: str = "Sample") -> str:
        """Create text explanation for a single prediction
        
        Args:
            sample: Feature vector (1D array)
            sample_name: Name of the sample for display
        
        Returns:
            String explanation
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Assuming sample is in the same data as SHAP was computed
        # Get top contributing features
        top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
        
        explanation = f"\\nExplanation for {sample_name}:\\n"
        explanation += "="*50 + "\\n"
        explanation += "Top contributing features:\\n"
        
        for i, idx in enumerate(top_idx, 1):
            feature_name = self.feature_names[idx]
            shap_value = shap_vals[0][idx]
            feature_value = sample[idx] if len(sample) > idx else "N/A"
            
            direction = "increases" if shap_value > 0 else "decreases"
            explanation += f"{i}. {feature_name}: {feature_value:.4f} ({direction} prediction by {abs(shap_value):.4f})\\n"
        
        return explanation


def analyze_stress_detection(model, X_train: np.ndarray, X_test: np.ndarray,
                            feature_names: list = None,
                            save_path: str = None) -> Dict:
    """Complete analysis pipeline for stress detection model
    
    Args:
        model: Fitted sklearn model
        X_train: Training features (for SHAP background)
        X_test: Test features
        feature_names: Feature names
        save_path: Path to save plots
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting comprehensive SHAP analysis...")
    
    # Create analyzer
    analyzer = SHAPAnalyzer(model, X_train[:200], feature_names)  # Use subset for background
    analyzer.create_explainer()
    analyzer.compute_shap_values(X_test)
    
    # Get results
    importance, names = analyzer.get_feature_importance()
    
    results = {
        'analyzer': analyzer,
        'feature_importance': importance[:10],  # Top 10
        'top_features': names[:10],
    }
    
    logger.info("Analysis complete!")
    
    return results

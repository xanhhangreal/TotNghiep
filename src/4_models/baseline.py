"""
Baseline models for stress detection
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, auc, roc_curve)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StressDetectionModel:
    """Base class for stress detection models"""
    
    def __init__(self, model_type: str = "random_forest", model_params: Dict = None):
        """
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'svm', 'decision_tree')
            model_params: Dictionary of model parameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(**self.model_params)
        elif self.model_type == "svm":
            return SVC(**self.model_params, probability=True)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            verbose: Print training info
        """
        # Handle NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data to train")
        
        if verbose:
            logger.info(f"Training {self.model_type} on {len(X_clean)} samples")
            logger.info(f"Number of features: {X_clean.shape[1]}")
            logger.info(f"Class distribution: {np.unique(y_clean, return_counts=True)}")
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X_clean)
        
        # Train model
        self.model.fit(X_normalized, y_clean)
        self.is_fitted = True
        
        if verbose:
            logger.info(f"Model training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress labels
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_normalized = self.scaler.transform(X)
        return self.model.predict(X_normalized)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_normalized = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_normalized)
        else:
            # For models without predict_proba (like SVM), estimate from decision function
            decision = self.model.decision_function(X_normalized)
            proba = 1 / (1 + np.exp(-decision))  # Sigmoid
            return np.column_stack([1 - proba, proba])
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, verbose: bool = True) -> Dict:
        """Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Label array
            cv: Number of folds
            verbose: Print results
        
        Returns:
            Dictionary of cross-validation scores
        """
        # Handle NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data for cross-validation")
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X_clean)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_weighted',
        }
        
        # Perform cross-validation
        cv_results = cross_validate(self.model, X_normalized, y_clean, 
                                   cv=cv, scoring=scoring, return_train_score=True)
        
        if verbose:
            logger.info(f"Cross-validation results ({cv}-fold):\"")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                logger.info(f"  {metric.upper()}: {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
                logger.info(f"    - Train: {np.mean(train_scores):.4f}")
        
        return cv_results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """Evaluate model on test data
        
        Args:
            X: Feature matrix
            y: True labels
            verbose: Print evaluation results
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True),
        }
        
        # ROC AUC (for binary classification)
        if len(np.unique(y)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
                fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
                metrics['roc_fpr'] = fpr
                metrics['roc_tpr'] = tpr
            except:
                metrics['roc_auc'] = np.nan
        
        if verbose:
            logger.info(f"\\nEvaluation Results:\"")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
            if 'roc_auc' in metrics and not np.isnan(metrics.get('roc_auc', np.nan)):
                logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            logger.info(f"\\nConfusion Matrix:\"")
            logger.info(f"{metrics['confusion_matrix']}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores
        
        Returns:
            Feature importance array or None if model doesn't support it
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, return absolute coefficients
            return np.abs(self.model.coef_[0])
        else:
            return None


def create_model(model_type: str = "random_forest", model_params: Dict = None) -> StressDetectionModel:
    """Factory function to create a stress detection model
    
    Args:
        model_type: Type of model
        model_params: Model parameters
    
    Returns:
        StressDetectionModel instance
    """
    return StressDetectionModel(model_type, model_params)


# Predefined model configurations
DEFAULT_MODELS = {
    'random_forest': {
        'params': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        }
    },
    'logistic_regression': {
        'params': {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs',
        }
    },
    'svm': {
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
        }
    },
    'decision_tree': {
        'params': {
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
        }
    }
}

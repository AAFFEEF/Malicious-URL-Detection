import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_metrics(self):
        """Calculates standard classification metrics."""
        y_pred = self.model.predict(self.X_test)
        
        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1-Score": f1_score(self.y_test, y_pred)
        }
        
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        return metrics

    def tpr_at_low_fpr(self, target_fpr=0.0001):
        """Calculates TPR at a specific low False Positive Rate."""
        # Get probabilities for the positive class
        if hasattr(self.model, "predict_proba"):
            y_probs = self.model.predict_proba(self.X_test)[:, 1]
        else:
            print("Model does not support predict_proba")
            return None
            
        fpr, tpr, thresholds = roc_curve(self.y_test, y_probs)
        
        # Find the index where FPR is closest to or less than target_fpr
        # We want the max TPR where FPR <= target_fpr
        valid_indices = np.where(fpr <= target_fpr)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[-1]
            print(f"\nTPR at FPR ~{fpr[idx]:.5f}: {tpr[idx]:.4f}")
            return tpr[idx]
        else:
            print("Could not find a threshold for the specified low FPR.")
            return 0.0

if __name__ == "__main__":
    # Test would require a trained model and data.
    pass

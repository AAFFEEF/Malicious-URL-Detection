import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, df, label_col='label'):
        self.df = df
        self.label_col = label_col
        self.selected_features = None

    def separate_features_target(self):
        X = self.df.drop(columns=[self.label_col, 'url']) # Drop non-numeric
        y = self.df[self.label_col]
        return X, y

    def remove_low_variance(self, threshold=0.0):
        """Removes features with variance lower than threshold."""
        # Simple implementation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Excluding label
        numeric_cols = [c for c in numeric_cols if c != self.label_col]
        
        keep = []
        for col in numeric_cols:
            if self.df[col].var() > threshold:
                keep.append(col)
        
        print(f"Features kept after variance check: {len(keep)}")
        return keep

    def select_with_model(self, X, y):
        """Uses Random Forest importance to select features."""
        print("Selecting features using Random Forest Importance...")
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        
        selected_indices = model.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        print(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        return self.df[self.selected_features + [self.label_col]]
    
    def optimization_algorithm_placeholder(self, algorithm="GA"):
        """
        Placeholder for Genetic Algorithm (GA), PSO, or HHO.
        In a full implementation, this would run an iterative search 
        to maximize a fitness function (e.g., accuracy).
        """
        print(f"Running {algorithm} optimization (Simulated)...")
        # For now, we return the existing features
        return self.df

if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        "url": ["a", "b", "c", "d"],
        "f1": [1, 2, 3, 4], 
        "f2": [0, 0, 0, 0], # Low variance
        "f3": [1, 0, 1, 0],
        "label": [0, 0, 1, 1]
    })
    selector = FeatureSelector(df)
    X, y = selector.separate_features_target()
    df_selected = selector.select_with_model(X, y)

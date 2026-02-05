import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os

class ModelTrainer:
    def __init__(self, df, label_col='label'):
        self.df = df
        self.label_col = label_col
        self.model = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self, features):
        """Splits data into Train and Test sets."""
        X = self.df[features]
        y = self.df[self.label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.X_test = X_test
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def train_rf(self, X_train, y_train):
        """Trains a Random Forest Model."""
        print("Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def train_xgboost(self, X_train, y_train):
        """Trains an XGBoost Model."""
        print("Training XGBoost...")
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)
        return self.model

    def train_deep_learning(self, X_train, y_train):
        """
        Placeholder for Deep Learning (CNN/BiLSTM).
        Requires TensorFlow/Keras.
        """
        print("Training Deep Learning Model (Simulated)...")
        # Example implementation:
        # model = Sequential()
        # model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(...)
        # self.model = model
        pass

    def save_model(self, filepath):
        """Saves the trained model to disk."""
        if self.model:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")

if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "label": np.random.randint(0, 2, 100)
    })
    trainer = ModelTrainer(df)
    X_train, X_test, y_train, y_test = trainer.prepare_data(["f1", "f2"])
    trainer.train_rf(X_train, y_train)
    trainer.save_model("../models/rf_model.pkl")

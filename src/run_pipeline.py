import pandas as pd
import os
import joblib
from src.data_ingestion import DataIngestion
from src.feature_extraction import FeatureExtractor
from src.feature_selection import FeatureSelector
from src.model_training import ModelTrainer
from src.evaluation import Evaluator

def main():
    print("=== Phase 1: Data Ingestion ===")
    # Initialize with no file to trigger dummy data generation
    ingestor = DataIngestion() 
    df = ingestor.load_data()
    ingestor.validate_schema()
    df = ingestor.normalize_urls()
    
    print("\n=== Phase 2: Feature Engineering ===")
    extractor = FeatureExtractor(df)
    df_features = extractor.run_all()
    
    print("\n=== Phase 3: Feature Selection ===")
    selector = FeatureSelector(df_features)
    X, y = selector.separate_features_target()
    
    # Simple variance threshold
    selected_cols = selector.remove_low_variance(threshold=0.01)
    
    # Model-based selection
    df_selected = selector.select_with_model(X[selected_cols], y)
    final_features = list(df_selected.columns)
    final_features.remove('label') # Keep only feature names
    
    print(f"Final Features: {len(final_features)}")
    
    print("\n=== Phase 4: Model Training ===")
    trainer = ModelTrainer(df_selected)
    X_train, X_test, y_train, y_test = trainer.prepare_data(final_features)
    
    # Train Random Forest
    model = trainer.train_rf(X_train, y_train)
    
    # Save Model and Feature List
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/rf_model.pkl")
    joblib.dump(final_features, "models/feature_list.pkl")
    
    print("\n=== Phase 5: Evaluation ===")
    evaluator = Evaluator(model, X_test, y_test)
    evaluator.evaluate_metrics()
    evaluator.tpr_at_low_fpr()
    
    print("\nPipeline Complete. Model ready for deployment.")

if __name__ == "__main__":
    main()

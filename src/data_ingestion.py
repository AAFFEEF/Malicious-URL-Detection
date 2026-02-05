import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

class DataIngestion:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from CSV or generates dummy data if file not found."""
        if self.file_path:
            try:
                self.data = pd.read_csv(self.file_path)
                print(f"Loaded data from {self.file_path}")
            except FileNotFoundError:
                print("File not found. Generating dummy data for demonstration.")
                self.data = self._generate_dummy_data()
        else:
            print("No file path provided. Generating dummy data.")
            self.data = self._generate_dummy_data()
        
        return self.data

    def _generate_dummy_data(self, n_samples=100):
        """Generates a synthetic dataset for testing."""
        phishing = [
            "http://paypal-secure-login.com/update",
            "http://apple-id-verify.info/login",
            "http://192.168.1.1/admin.html",
            "http://google-drive-secure.xyz/download",
            "http://signin.ebay.com.account-update.tk"
        ]
        legit = [
            "https://www.google.com",
            "https://www.amazon.com/products",
            "https://github.com/login",
            "https://stackoverflow.com/questions",
            "https://en.wikipedia.org/wiki/Main_Page"
        ]
        
        # Randomly sample
        urls = np.random.choice(phishing + legit, n_samples)
        labels = [1 if u in phishing else 0 for u in urls] # 1 = Malicious
        
        return pd.DataFrame({"url": urls, "label": labels})

    def validate_schema(self):
        """
        Performs rigorous validation:
        1. Checks for required columns.
        2. checks for null/missing values.
        3. Checks for duplicates.
        4. Validates data types.
        """
        required_cols = ["url", "label"]
        
        # 1. Column Existence
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")
        
        # 2. Missing Values
        if self.data[required_cols].isnull().any().any():
            logging = f"Found missing values:\n{self.data[required_cols].isnull().sum()}"
            print(logging)
            self.data.dropna(subset=required_cols, inplace=True)
            print("Dropped rows with missing values.")

        # 3. Duplicates
        if self.data.duplicated(subset=['url']).any():
            dup_count = self.data.duplicated(subset=['url']).sum()
            print(f"Found {dup_count} duplicate URLs. Removing them...")
            self.data.drop_duplicates(subset=['url'], inplace=True)

        # 4. Data Types
        if not pd.api.types.is_string_dtype(self.data['url']):
            raise ValueError("Column 'url' must be string type.")
        
        # Verify labels are binary (if we are strictly doing binary classification)
        unique_labels = self.data['label'].unique()
        if not all(l in [0, 1] for l in unique_labels):
             print(f"Warning: Labels contain values other than 0 and 1: {unique_labels}")

        print("Data validation passed (Schema, Nulls, Duplicates, Types checked).")

    def normalize_urls(self):
        """Basic normalization: lowercase, strip."""
        if self.data is not None:
            self.data['url'] = self.data['url'].str.lower().str.strip()
            # Remove www.
            self.data['url'] = self.data['url'].str.replace('www.', '', regex=False)
            print("URL normalization complete.")
        return self.data

if __name__ == "__main__":
    # Test the module
    ingestor = DataIngestion()
    df = ingestor.load_data()
    ingestor.validate_schema()
    df = ingestor.normalize_urls()
    print(df.head())

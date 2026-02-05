import pandas as pd
import numpy as np
import re
import math
import tldextract
from urllib.parse import urlparse
import socket
import requests
from collections import Counter

class FeatureExtractor:
    def __init__(self, df):
        self.df = df

    def _shannon_entropy(self, text):
        """Calculates Shannon Entropy of a string."""
        if not text:
            return 0
        counts = Counter(text)
        frequencies = [i / len(text) for i in counts.values()]
        return -sum(f * math.log2(f) for f in frequencies)

    def extract_lexical_features(self):
        """Extracts text-based properties of the URL."""
        print("Extracting Lexical Features...")
        
        # URL Length
        self.df['url_length'] = self.df['url'].apply(len)
        
        # Hostname Length
        self.df['hostname_length'] = self.df['url'].apply(lambda x: len(urlparse(x).netloc))
        
        # Path Length
        self.df['path_length'] = self.df['url'].apply(lambda x: len(urlparse(x).path))
        
        # Special Character Counts
        chars = ['@', '?', '-', '=', '.', '%', '+', '$', '!', '*', ',', '//']
        for char in chars:
            col_name = f"count_{char}"
            self.df[col_name] = self.df['url'].apply(lambda x: x.count(char))
            
        # Digit Count
        self.df['count_digits'] = self.df['url'].apply(lambda x: sum(c.isdigit() for c in x))
        
        # Shannon Entropy
        self.df['url_entropy'] = self.df['url'].apply(self._shannon_entropy)
        
        # TLD Count (Subdomains)
        self.df['subdomain_count'] = self.df['url'].apply(lambda x: len(tldextract.extract(x).subdomain.split('.')) if tldextract.extract(x).subdomain else 0)
        
        # IP Address in URL
        ip_pattern = r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5]))'
        self.df['has_ip'] = self.df['url'].apply(lambda x: 1 if re.search(ip_pattern, x) else 0)
        
        # Suspicious Words
        suspicious = ['login', 'signin', 'bank', 'verify', 'update', 'secure', 'ebay', 'paypal']
        self.df['has_suspicious_words'] = self.df['url'].apply(lambda x: 1 if any(s in x for s in suspicious) else 0)
        
        return self.df

    def extract_host_features(self):
        """
        Extracts host-based features (WHOIS, IP Geolocation).
        NOTE: Real-time WHOIS/GeoIP is slow and requires internet.
        For this implementation, we will mock these or check for specific patterns.
        """
        print("Extracting Host-Based Features (Mocked)...")
        # In a real scenario, you would use python-whois library here.
        # Example:
        # try:
        #     w = whois.whois(domain)
        #     creation_date = w.creation_date
        #     ...
        # except:
        #     ...
        
        # Mocking Domain Age (random for demonstration)
        # 1 = New (<6 months), 0 = Old
        self.df['domain_age_days'] = np.random.randint(0, 365, size=len(self.df)) 
        self.df['is_domain_new'] = (self.df['domain_age_days'] < 180).astype(int)
        
        return self.df

    def extract_content_features(self):
        """
        Extracts content-based features (HTML analysis).
        NOTE: Requires fetching the URL. Skipped/Mocked for safety and speed in this demo.
        """
        print("Extracting Content-Based Features (Mocked)...")
        
        # Mocking HTML features
        # presence of iframe, excessive script tags, etc.
        self.df['has_iframe'] = np.random.randint(0, 2, size=len(self.df))
        self.df['num_scripts'] = np.random.randint(0, 10, size=len(self.df))
        
        return self.df

    def run_all(self):
        self.extract_lexical_features()
        self.extract_host_features()
        self.extract_content_features()
        return self.df

if __name__ == "__main__":
    # Test
    data = {"url": ["http://google.com", "http://192.168.1.1/login"], "label": [0, 1]}
    df = pd.DataFrame(data)
    extractor = FeatureExtractor(df)
    df_features = extractor.run_all()
    print(df_features.columns)

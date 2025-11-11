"""
Data Preprocessing Module
Handles cleaning, missing values, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPreprocessor:
    """Preprocessor for Mental Health Tech Survey data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.selected_features = []
        
    def select_important_features(self, df):
        """Select most relevant features for clustering"""
        print("\nSelecting important features...")
        
        # Core features with good information and reasonable missing values
        important_cols = [
            'Are you self-employed?',
            'How many employees does your company or organization have?',
            'Is your employer primarily a tech company/organization?',
            'Does your employer provide mental health benefits as part of healthcare coverage?',
            'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
            'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
            'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
            'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
            'Do you think that discussing a physical health issue with your employer would have negative consequences?',
            'Would you feel comfortable discussing a mental health disorder with your coworkers?',
            'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
            'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?',
            'Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?',
            'Do you know local or online resources to seek help for a mental health disorder?',
            'Have you been diagnosed with a mental health condition by a medical professional?',
            'Do you believe you have a mental health condition?',
            'Have you sought treatment for a mental health issue from a mental health professional?',
            'Does your employer provide mental health benefits as part of healthcare coverage?.1',
            'Do you know the options for mental health care available under your employer-provided coverage?',
            'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?.1',
            'Does your employer offer resources to learn more about mental health concerns and options for seeking help?.1',
            'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
            'Would you bring up a mental health issue with a potential employer in an interview?',
            'Do you feel that being identified as a person with a mental health issue would hurt your career?',
            'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?',
            'How willing would you be to share with friends and family that you have a mental illness?',
            'What is your age?',
            'What is your gender?',
        ]
        
        # Filter to existing columns
        existing_cols = [col for col in important_cols if col in df.columns]
        
        print(f"Selected {len(existing_cols)} important features")
        self.selected_features = existing_cols
        
        return df[existing_cols].copy()
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately"""
        print("\nHandling missing values...")
        
        # Identify column types
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  Numerical columns: {len(self.numerical_cols)}")
        print(f"  Categorical columns: {len(self.categorical_cols)}")
        
        # Strategy 1: Drop rows with too many missing values (>50% missing)
        missing_pct = df.isnull().sum(axis=1) / len(df.columns)
        df_clean = df[missing_pct < 0.5].copy()
        print(f"  Dropped {len(df) - len(df_clean)} rows with >50% missing values")
        
        # Strategy 2: Impute numerical columns with median
        if len(self.numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df_clean[self.numerical_cols] = num_imputer.fit_transform(df_clean[self.numerical_cols])
        
        # Strategy 3: Impute categorical with most frequent or 'Unknown'
        if len(self.categorical_cols) > 0:
            for col in self.categorical_cols:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        print(f"  Final shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            # Binary mapping for yes/no questions
            if df_encoded[col].nunique() <= 3:
                # Common patterns
                mapping = {
                    'Yes': 1, 'No': 0, 'Maybe': 0.5, 'I don\'t know': 0.5,
                    'Unknown': 0, 'Not sure': 0.5,
                    '1': 1, '0': 0, 1: 1, 0: 0
                }
                
                # Try to map, otherwise use label encoding
                try:
                    df_encoded[col] = df_encoded[col].map(mapping)
                    df_encoded[col] = df_encoded[col].fillna(0.5)
                except:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
            else:
                # Multi-category: use label encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"  Encoded {len(self.categorical_cols)} categorical columns")
        return df_encoded
    
    def scale_features(self, df):
        """Scale all features to standard scale"""
        print("\nScaling features...")
        
        df_scaled = df.copy()
        feature_names = df_scaled.columns.tolist()
        
        # Standardize all features
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_scaled),
            columns=feature_names,
            index=df_scaled.index
        )
        
        print(f"  Scaled {len(feature_names)} features")
        return df_scaled
    
    def fit_transform(self, df):
        """Complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        print(f"Input shape: {df.shape}")
        
        # Step 1: Select important features
        df_selected = self.select_important_features(df)
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df_selected)
        
        # Step 3: Encode categorical variables
        df_encoded = self.encode_categorical(df_clean)
        
        # Step 4: Scale features
        df_final = self.scale_features(df_encoded)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Output shape: {df_final.shape}")
        print(f"Features retained: {list(df_final.columns[:5])}... ({len(df_final.columns)} total)")
        
        return df_final

def main():
    """Test preprocessing"""
    # Load data
    df = pd.read_csv('data/mental_health_tech_2016.csv')
    
    # Preprocess
    preprocessor = MentalHealthPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # Save
    df_processed.to_csv('outputs/processed_data.csv', index=False)
    print(f"\n✓ Processed data saved to outputs/processed_data.csv")
    
    return df_processed

if __name__ == "__main__":
    main()


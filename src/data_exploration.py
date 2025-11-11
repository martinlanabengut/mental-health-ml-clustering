"""
Data Exploration Module
Performs initial exploratory data analysis on the Mental Health dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(file_path='data/mental_health_tech_2016.csv'):
    """Load the mental health dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-"*80)
    print("DATA TYPES")
    print("-"*80)
    print(df.dtypes.value_counts())
    
    print("\n" + "-"*80)
    print("MISSING VALUES")
    print("-"*80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n{len(missing_df)} columns with missing values:")
        print(missing_df.head(20).to_string(index=False))
    else:
        print("\n✓ No missing values found!")
    
    return missing_df

def column_analysis(df):
    """Analyze column names and types"""
    print("\n" + "="*80)
    print("COLUMN ANALYSIS")
    print("="*80)
    
    print(f"\nTotal columns: {len(df.columns)}")
    print("\nFirst 20 columns:")
    for i, col in enumerate(df.columns[:20], 1):
        dtype = df[col].dtype
        unique = df[col].nunique()
        print(f"{i:2d}. {col[:60]:<60} | {str(dtype):<10} | {unique:>5} unique")
    
    if len(df.columns) > 20:
        print(f"\n... and {len(df.columns) - 20} more columns")

def save_summary(df, output_dir='outputs'):
    """Save data summary to file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    summary = df.describe(include='all').T
    summary.to_csv(f'{output_dir}/data_summary.csv')
    print(f"\n✓ Summary saved to {output_dir}/data_summary.csv")
    
    # Column info
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Unique_Values': [df[col].nunique() for col in df.columns],
        'Missing': df.isnull().sum().values,
        'Missing_Pct': (df.isnull().sum() / len(df) * 100).values
    })
    col_info.to_csv(f'{output_dir}/column_info.csv', index=False)
    print(f"✓ Column info saved to {output_dir}/column_info.csv")

def main():
    """Main execution"""
    # Load data
    df = load_data()
    
    # Basic info
    basic_info(df)
    
    # Column analysis
    column_analysis(df)
    
    # Save summary
    save_summary(df)
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the column_info.csv file")
    print("2. Identify which columns are useful for clustering")
    print("3. Plan preprocessing strategy")
    
    return df

if __name__ == "__main__":
    main()


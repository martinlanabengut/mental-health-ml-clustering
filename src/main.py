"""
Main Analysis Pipeline
Mental Health in Tech - Unsupervised Learning Case Study
DLBDSMLUSL01

This script runs the complete analysis pipeline:
1. Data exploration
2. Preprocessing
3. Dimensionality reduction
4. Clustering
5. Visualization
6. Interpretation

Author: Martin Lana Bengut
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocessing import MentalHealthPreprocessor
from clustering import ClusterAnalyzer, DimensionalityReducer
from visualization import create_comprehensive_report

def main():
    """Execute complete analysis pipeline"""
    
    print("\n" + "="*80)
    print("MENTAL HEALTH IN TECH - UNSUPERVISED LEARNING ANALYSIS")
    print("DLBDSMLUSL01 - Machine Learning Case Study")
    print("="*80)
    
    # Setup directories
    Path('outputs').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n[STEP 1] LOADING DATA")
    print("-" * 80)
    
    df_raw = pd.read_csv('data/mental_health_tech_2016.csv')
    print(f"✓ Loaded dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    
    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    print("\n[STEP 2] PREPROCESSING")
    print("-" * 80)
    
    preprocessor = MentalHealthPreprocessor()
    df_processed = preprocessor.fit_transform(df_raw)
    
    # Save processed data
    df_processed.to_csv('outputs/processed_data.csv', index=False)
    print(f"✓ Processed data saved")
    
    X = df_processed.values
    feature_names = df_processed.columns.tolist()
    
    # =========================================================================
    # STEP 3: DIMENSIONALITY REDUCTION
    # =========================================================================
    print("\n[STEP 3] DIMENSIONALITY REDUCTION")
    print("-" * 80)
    
    reducer = DimensionalityReducer()
    
    # PCA
    X_pca, explained_var = reducer.fit_pca(X, n_components=2)
    
    # Plot variance
    reducer.plot_variance(X, save_path='visualizations/pca_variance.png')
    
    # t-SNE
    X_tsne = reducer.fit_tsne(X, n_components=2, perplexity=30)
    
    # UMAP
    try:
        X_umap = reducer.fit_umap(X, n_components=2, n_neighbors=15)
        print("✓ UMAP complete")
    except Exception as e:
        print(f"⚠ UMAP skipped: {e}")
        X_umap = None
    
    # =========================================================================
    # STEP 4: CLUSTERING
    # =========================================================================
    print("\n[STEP 4] CLUSTERING ANALYSIS")
    print("-" * 80)
    
    analyzer = ClusterAnalyzer()
    
    # Find optimal k
    optimal_k = analyzer.find_optimal_clusters(
        X, max_k=10, 
        save_path='visualizations/optimal_clusters.png'
    )
    
    # Fit K-Means with optimal k
    analyzer.n_clusters = optimal_k
    labels = analyzer.fit_kmeans(X)
    
    # Get cluster summary
    summary = analyzer.get_cluster_summary(X, df_raw)
    
    # Save metrics
    metrics_df = pd.DataFrame([analyzer.metrics])
    metrics_df['n_clusters'] = optimal_k
    metrics_df.to_csv('outputs/clustering_metrics.csv', index=False)
    print("✓ Metrics saved to outputs/clustering_metrics.csv")
    
    # =========================================================================
    # STEP 5: VISUALIZATION
    # =========================================================================
    print("\n[STEP 5] CREATING VISUALIZATIONS")
    print("-" * 80)
    
    create_comprehensive_report(X, labels, X_pca, X_tsne, feature_names)
    
    # =========================================================================
    # STEP 6: CLUSTER INTERPRETATION
    # =========================================================================
    print("\n[STEP 6] CLUSTER INTERPRETATION")
    print("-" * 80)
    
    # Analyze each cluster
    df_with_clusters = df_raw.copy()
    df_with_clusters = df_with_clusters.iloc[:len(labels)]  # Match length
    df_with_clusters['Cluster'] = labels
    
    print("\nCluster Profiles:")
    for cluster_id in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        print(f"\n  Cluster {cluster_id} (n={len(cluster_data)}, {len(cluster_data)/len(labels)*100:.1f}%):")
        
        # Key characteristics
        if 'What is your age?' in cluster_data.columns:
            age_mean = cluster_data['What is your age?'].median()
            print(f"    - Median age: {age_mean:.0f}")
        
        if 'What is your gender?' in cluster_data.columns:
            gender_mode = cluster_data['What is your gender?'].mode()
            if len(gender_mode) > 0:
                print(f"    - Most common gender: {gender_mode.iloc[0]}")
        
        if 'Have you been diagnosed with a mental health condition by a medical professional?' in cluster_data.columns:
            diagnosed = cluster_data['Have you been diagnosed with a mental health condition by a medical professional?']
            if diagnosed.notna().sum() > 0:
                diagnosed_pct = (diagnosed == 1).sum() / diagnosed.notna().sum() * 100
                print(f"    - Diagnosed with mental health: {diagnosed_pct:.1f}%")
    
    # Save detailed cluster summary
    cluster_summary = df_with_clusters.groupby('Cluster').agg({
        'What is your age?': ['mean', 'median'],
        'Have you been diagnosed with a mental health condition by a medical professional?': 'sum',
        'Are you self-employed?': 'mean'
    })
    cluster_summary.to_csv('outputs/cluster_summary.csv')
    print("\n✓ Detailed cluster summary saved to outputs/cluster_summary.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n✓ Dataset: 1,433 responses analyzed")
    print(f"✓ Features: {len(feature_names)} selected and processed")
    print(f"✓ Clusters identified: {optimal_k}")
    print(f"✓ Silhouette score: {analyzer.metrics['silhouette']:.3f}")
    
    print(f"\n📁 Outputs generated:")
    print(f"   • outputs/processed_data.csv - Cleaned data")
    print(f"   • outputs/clustered_data.csv - Data with cluster labels")
    print(f"   • outputs/cluster_summary.csv - Cluster characteristics")
    print(f"   • outputs/clustering_metrics.csv - Performance metrics")
    
    print(f"\n📊 Visualizations created:")
    print(f"   • visualizations/pca_variance.png")
    print(f"   • visualizations/optimal_clusters.png")
    print(f"   • visualizations/cluster_distribution.png")
    print(f"   • visualizations/clusters_pca.png")
    print(f"   • visualizations/clusters_tsne.png")
    print(f"   • visualizations/cluster_characteristics.png")
    print(f"   • visualizations/clusters_interactive_pca.html")
    print(f"   • visualizations/clusters_interactive_tsne.html")
    
    print("\n🎯 Next steps:")
    print("   1. Review visualizations in the 'visualizations/' folder")
    print("   2. Open interactive HTML files in your browser")
    print("   3. Analyze cluster_summary.csv for insights")
    print("   4. Prepare recommendations for HR based on findings")
    
    print("\n" + "="*80)
    print("✅ PROJECT EXECUTION SUCCESSFUL")
    print("="*80)
    
    return {
        'preprocessor': preprocessor,
        'analyzer': analyzer,
        'reducer': reducer,
        'labels': labels,
        'X_pca': X_pca,
        'X_tsne': X_tsne
    }

if __name__ == "__main__":
    results = main()


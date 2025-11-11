"""
Clustering Module
Implements multiple clustering algorithms and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ClusterAnalyzer:
    """Clustering analysis for Mental Health data"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = None
        self.labels = None
        self.metrics = {}
        
    def find_optimal_clusters(self, X, max_k=10, save_path=None):
        """Find optimal number of clusters using elbow method and silhouette"""
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            
            print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.3f}")
        
        # Plot
        if save_path:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Elbow plot
            ax1.plot(k_range, inertias, 'bo-')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax1.set_ylabel('Inertia', fontsize=12)
            ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # Silhouette plot
            ax2.plot(k_range, silhouettes, 'ro-')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax2.set_ylabel('Silhouette Score', fontsize=12)
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            # Mark optimal k (highest silhouette)
            optimal_k = k_range[np.argmax(silhouettes)]
            ax2.axvline(optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved optimal clusters plot to {save_path}")
        
        # Recommend optimal k
        optimal_k = k_range[np.argmax(silhouettes)]
        print(f"\n✓ Recommended number of clusters: {optimal_k}")
        
        return optimal_k
    
    def fit_kmeans(self, X):
        """Fit K-Means clustering"""
        print(f"\nFitting K-Means with k={self.n_clusters}...")
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(X)
        
        # Calculate metrics
        self.metrics['silhouette'] = silhouette_score(X, self.labels)
        self.metrics['calinski_harabasz'] = calinski_harabasz_score(X, self.labels)
        self.metrics['davies_bouldin'] = davies_bouldin_score(X, self.labels)
        
        print(f"  Silhouette Score: {self.metrics['silhouette']:.3f}")
        print(f"  Calinski-Harabasz: {self.metrics['calinski_harabasz']:.2f}")
        print(f"  Davies-Bouldin: {self.metrics['davies_bouldin']:.3f}")
        
        # Cluster sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n  Cluster sizes:")
        for cluster, count in zip(unique, counts):
            pct = (count / len(self.labels)) * 100
            print(f"    Cluster {cluster}: {count} samples ({pct:.1f}%)")
        
        return self.labels
    
    def evaluate_metrics(self):
        """Return evaluation metrics"""
        return self.metrics
    
    def get_cluster_summary(self, X, original_df=None):
        """Analyze cluster characteristics"""
        print("\nAnalyzing cluster characteristics...")
        
        df_clustered = pd.DataFrame(X)
        df_clustered['Cluster'] = self.labels
        
        # Statistical summary per cluster
        summary = df_clustered.groupby('Cluster').agg(['mean', 'std', 'count'])
        
        # If original data available, get more meaningful names
        if original_df is not None and len(original_df) == len(self.labels):
            df_with_clusters = original_df.copy()
            df_with_clusters['Cluster'] = self.labels
            
            # Save clustered data
            df_with_clusters.to_csv('outputs/clustered_data.csv', index=False)
            print("✓ Saved clustered data to outputs/clustered_data.csv")
        
        return summary

class DimensionalityReducer:
    """Dimensionality reduction for visualization"""
    
    def __init__(self):
        self.pca = None
        self.tsne = None
        self.umap_model = None
        
    def fit_pca(self, X, n_components=2):
        """Apply PCA"""
        print(f"\nApplying PCA (n_components={n_components})...")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        # Explained variance
        explained_var = self.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        print(f"  PC1 explains: {explained_var[0]*100:.1f}%")
        print(f"  PC2 explains: {explained_var[1]*100:.1f}%")
        print(f"  Total: {cumsum_var[1]*100:.1f}%")
        
        return X_pca, explained_var
    
    def fit_tsne(self, X, n_components=2, perplexity=30):
        """Apply t-SNE"""
        print(f"\nApplying t-SNE (perplexity={perplexity})...")
        print("  This may take a few minutes...")
        
        self.tsne = TSNE(n_components=n_components, random_state=42, 
                         perplexity=perplexity, max_iter=1000)
        X_tsne = self.tsne.fit_transform(X)
        
        print("  ✓ t-SNE complete")
        return X_tsne
    
    def fit_umap(self, X, n_components=2, n_neighbors=15):
        """Apply UMAP"""
        print(f"\nApplying UMAP (n_neighbors={n_neighbors})...")
        
        self.umap_model = umap.UMAP(n_components=n_components, random_state=42,
                                     n_neighbors=n_neighbors)
        X_umap = self.umap_model.fit_transform(X)
        
        print("  ✓ UMAP complete")
        return X_umap
    
    def plot_variance(self, X, save_path=None):
        """Plot PCA explained variance"""
        print("\nAnalyzing PCA variance...")
        
        pca_full = PCA(random_state=42)
        pca_full.fit(X)
        
        explained_var = pca_full.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('PCA - Variance per Component', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, min(20, len(explained_var) + 1))
        
        # Cumulative variance
        ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
        ax2.axhline(0.8, color='r', linestyle='--', label='80% variance')
        ax2.axhline(0.9, color='g', linestyle='--', label='90% variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('PCA - Cumulative Variance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, min(20, len(cumsum_var) + 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved PCA variance plot to {save_path}")
        else:
            plt.show()
        
        # Print summary
        n_80 = np.argmax(cumsum_var >= 0.8) + 1
        n_90 = np.argmax(cumsum_var >= 0.9) + 1
        print(f"\n  Components needed for 80% variance: {n_80}")
        print(f"  Components needed for 90% variance: {n_90}")

def main():
    """Test clustering"""
    # Load processed data
    df_processed = pd.read_csv('outputs/processed_data.csv')
    X = df_processed.values
    
    # Find optimal k
    analyzer = ClusterAnalyzer()
    optimal_k = analyzer.find_optimal_clusters(X, max_k=10, 
                                               save_path='visualizations/optimal_clusters.png')
    
    # Fit with optimal k
    analyzer.n_clusters = optimal_k
    labels = analyzer.fit_kmeans(X)
    
    # Dimensionality reduction
    reducer = DimensionalityReducer()
    reducer.plot_variance(X, save_path='visualizations/pca_variance.png')
    
    print("\n✓ Clustering analysis complete")
    
    return analyzer, reducer

if __name__ == "__main__":
    main()


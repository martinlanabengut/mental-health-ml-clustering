"""
Visualization Module
Creates plots for cluster analysis and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_clusters_2d(X_reduced, labels, title="Cluster Visualization", 
                     save_path=None, method='PCA'):
    """Plot clusters in 2D space"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                        c=labels, cmap='tab10', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add cluster centers if available
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        center = X_reduced[mask].mean(axis=0)
        ax.scatter(center[0], center[1], c='red', s=300, 
                  marker='X', edgecolors='black', linewidth=2,
                  label=f'Cluster {label} center')
    
    ax.set_xlabel(f'{method} Component 1', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {method} cluster plot to {save_path}")
    else:
        plt.show()

def plot_interactive_clusters(X_reduced, labels, method='PCA', save_path=None):
    """Create interactive 3D plot with Plotly"""
    
    df_plot = pd.DataFrame({
        f'{method}1': X_reduced[:, 0],
        f'{method}2': X_reduced[:, 1],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter(df_plot, x=f'{method}1', y=f'{method}2', color='Cluster',
                     title=f'Interactive Cluster Visualization ({method})',
                     color_discrete_sequence=px.colors.qualitative.Set1,
                     hover_data={'Cluster': True})
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        width=1000, height=700,
        font=dict(size=12),
        title_font_size=16
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved interactive {method} plot to {save_path}")
    else:
        fig.show()

def plot_cluster_distribution(labels, save_path=None):
    """Plot distribution of samples across clusters"""
    
    unique, counts = np.unique(labels, return_counts=True)
    percentages = (counts / len(labels)) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = plt.cm.tab10(unique / unique.max())
    bars = ax1.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(unique)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=[f'Cluster {i}' for i in unique], 
           autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Cluster Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cluster distribution to {save_path}")
    else:
        plt.show()

def plot_feature_importance_per_cluster(X, labels, feature_names, top_n=10, save_path=None):
    """Plot top features distinguishing each cluster"""
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels
    
    n_clusters = len(np.unique(labels))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(n_clusters, 4)):
        ax = axes[i]
        
        # Calculate mean values for this cluster vs others
        cluster_mean = df[df['Cluster'] == i].drop('Cluster', axis=1).mean()
        others_mean = df[df['Cluster'] != i].drop('Cluster', axis=1).mean()
        
        # Difference (how this cluster differs)
        diff = cluster_mean - others_mean
        top_features = diff.abs().nlargest(top_n)
        
        # Plot
        colors = ['green' if diff[feat] > 0 else 'red' for feat in top_features.index]
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, [diff[feat] for feat in top_features.index], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat[:40] + '...' if len(feat) > 40 else feat 
                            for feat in top_features.index], fontsize=9)
        ax.set_xlabel('Difference from Other Clusters', fontsize=10)
        ax.set_title(f'Cluster {i} - Distinguishing Features', 
                    fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved feature importance plot to {save_path}")
    else:
        plt.show()

def create_comprehensive_report(X, labels, X_pca, X_tsne, feature_names, 
                               output_dir='visualizations'):
    """Create all visualizations"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Cluster distribution
    plot_cluster_distribution(labels, save_path=f'{output_dir}/cluster_distribution.png')
    
    # 2. PCA clusters
    plot_clusters_2d(X_pca, labels, 
                    title='Clusters in PCA Space',
                    save_path=f'{output_dir}/clusters_pca.png',
                    method='PCA')
    
    # 3. t-SNE clusters
    plot_clusters_2d(X_tsne, labels,
                    title='Clusters in t-SNE Space', 
                    save_path=f'{output_dir}/clusters_tsne.png',
                    method='t-SNE')
    
    # 4. Feature importance
    plot_feature_importance_per_cluster(X, labels, feature_names,
                                       save_path=f'{output_dir}/cluster_characteristics.png')
    
    # 5. Interactive plots
    plot_interactive_clusters(X_pca, labels, method='PCA',
                             save_path=f'{output_dir}/clusters_interactive_pca.html')
    plot_interactive_clusters(X_tsne, labels, method='t-SNE',
                             save_path=f'{output_dir}/clusters_interactive_tsne.html')
    
    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("  • cluster_distribution.png")
    print("  • clusters_pca.png")
    print("  • clusters_tsne.png")
    print("  • cluster_characteristics.png")
    print("  • clusters_interactive_pca.html (open in browser)")
    print("  • clusters_interactive_tsne.html (open in browser)")

def main():
    """Test visualization"""
    # Load processed data
    df_processed = pd.read_csv('outputs/processed_data.csv')
    X = df_processed.values
    feature_names = df_processed.columns.tolist()
    
    # Load cluster labels (if available)
    try:
        df_clustered = pd.read_csv('outputs/clustered_data.csv')
        labels = df_clustered['Cluster'].values
    except:
        print("No cluster labels found. Run clustering first.")
        return
    
    # Apply dimensionality reduction
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Create visualizations
    create_comprehensive_report(X, labels, X_pca, X_tsne, feature_names)

if __name__ == "__main__":
    main()


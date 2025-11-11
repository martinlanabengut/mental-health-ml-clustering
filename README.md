# Mental Health in Technology - Unsupervised Learning Case Study
**DLBDSMLUSL01 – Machine Learning – Unsupervised Learning and Feature Engineering**

Author: Martin Lana Bengut  
Date: November 2025

---

## 🎯 Project Overview

This case study analyzes mental health patterns among technology workers using unsupervised learning techniques. The goal is to help HR departments identify employee groups and develop targeted mental health interventions.

## 📊 Key Results

- **Dataset:** 1,433 survey responses → 1,146 after cleaning
- **Features:** 63 original → 22 selected features
- **Clusters Identified:** 2 distinct groups
- **Silhouette Score:** 0.121
- **Variance Explained (PCA 2D):** 28.4%

### Cluster Distribution:
- **Cluster 0:** 618 employees (53.9%)
- **Cluster 1:** 528 employees (46.1%)

## 🚀 Quick Start

### Installation

```bash
cd "/Users/martinlanabengut/Desktop/Mental Health ML Project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Activate environment
source venv/bin/activate

# Run complete pipeline
python src/main.py
```

This will:
1. Load and explore data
2. Preprocess and clean
3. Apply dimensionality reduction (PCA, t-SNE, UMAP)
4. Perform clustering analysis
5. Generate all visualizations
6. Save results to `outputs/` and `visualizations/`

## 📁 Project Structure

```
Mental Health ML Project/
├── data/
│   └── mental_health_tech_2016.csv          # Raw dataset (1.1 MB)
├── src/
│   ├── data_exploration.py                   # Initial EDA
│   ├── preprocessing.py                      # Data cleaning
│   ├── clustering.py                         # Clustering algorithms
│   ├── visualization.py                      # Plotting functions
│   └── main.py                               # Main pipeline
├── outputs/
│   ├── processed_data.csv                    # Cleaned data
│   ├── clustered_data.csv                    # Data with cluster labels
│   ├── cluster_summary.csv                   # Cluster statistics
│   └── clustering_metrics.csv                # Performance metrics
├── visualizations/
│   ├── pca_variance.png                      # PCA explained variance
│   ├── optimal_clusters.png                  # Elbow + Silhouette
│   ├── cluster_distribution.png              # Cluster sizes
│   ├── clusters_pca.png                      # PCA visualization
│   ├── clusters_tsne.png                     # t-SNE visualization
│   ├── cluster_characteristics.png           # Feature importance
│   ├── clusters_interactive_pca.html         # Interactive PCA
│   └── clusters_interactive_tsne.html        # Interactive t-SNE
├── CASE_STUDY_DOCUMENTATION.md               # Full report
├── requirements.txt                          # Dependencies
└── README.md                                 # This file
```

## 🔬 Methodology

### 1. Data Preprocessing
- Selected 22 most relevant features
- Handled missing values (dropped rows with >50% missing)
- Encoded categorical variables
- Standardized all features

### 2. Dimensionality Reduction
- **PCA:** Explained 28.4% variance in 2D
- **t-SNE:** Non-linear projection for visualization
- **UMAP:** Alternative manifold learning

### 3. Clustering
- Tested K-Means with k=2 to 10
- Optimal clusters: 2 (based on silhouette score)
- Evaluation metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin

### 4. Visualization
- Static plots (PNG)
- Interactive plots (HTML)
- Feature importance analysis

## 📊 Dataset

**Mental Health in Tech Survey 2016**
- **Source:** https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016
- **Size:** 1,433 responses × 63 features
- **Topics:**
  - Employment details
  - Company mental health policies
  - Personal mental health history
  - Attitudes toward mental health
  - Treatment experiences

## 📈 Visualizations

All visualizations are saved in `visualizations/`:

1. **pca_variance.png** - Shows how much variance each PC captures
2. **optimal_clusters.png** - Elbow and silhouette analysis
3. **cluster_distribution.png** - Bar and pie charts of cluster sizes
4. **clusters_pca.png** - 2D PCA projection with clusters
5. **clusters_tsne.png** - 2D t-SNE projection with clusters
6. **cluster_characteristics.png** - Features distinguishing each cluster
7. **Interactive HTML plots** - Open in browser for exploration

## 🛠️ Technologies

- **Python 3.10+**
- **pandas** - Data manipulation
- **scikit-learn** - ML algorithms
- **matplotlib, seaborn** - Static plots
- **plotly** - Interactive visualizations
- **umap-learn** - UMAP algorithm

## 📝 Documentation

Complete case study documentation: `CASE_STUDY_DOCUMENTATION.md`

Includes:
- Business context
- Data exploration
- Preprocessing decisions
- Clustering methodology
- Results interpretation
- HR recommendations

## 🔗 Links

- **GitHub Repository:** https://github.com/martinlanabengut/mental-health-ml-clustering
- **Dataset Source:** https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016

## 📧 Contact

Martin Lana Bengut  
DLBDSMLUSL01 – Machine Learning Case Study

---

**Status:** ✅ Analysis Complete


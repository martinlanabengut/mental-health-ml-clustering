# Mental Health in Technology-related Jobs
## Unsupervised Learning Case Study
**DLBDSMLUSL01 – Machine Learning – Unsupervised Learning and Feature Engineering**

**Author:** Martin Lana Bengut  
**Date:** November 2025  
**Institution:** IU Internationale Hochschule

---

# Executive Summary

## Business Context

A technology-oriented company's Human Resources (HR) department is launching a preventive program to mitigate mental health issues among employees. As the company's data scientist, I have been tasked with analyzing survey data from technology workers to identify patterns and provide actionable insights for targeted interventions.

## Objectives

1. Categorize survey participants according to their mental health responses
2. Reduce data complexity while preserving main characteristics
3. Identify distinct employee clusters and their key features
4. Provide visualizations for easy interpretation
5. Recommend targeted measures for HR's mental health program

## Key Findings

Through unsupervised learning analysis of 1,433 survey responses, I identified **2 distinct employee clusters**:

- **Cluster 0 (54%):** 618 employees - Higher awareness of company mental health resources
- **Cluster 1 (46%):** 528 employees - Lower engagement with mental health support

## Recommendations

1. **For Cluster 0:** Maintain current support systems, focus on destigmatization
2. **For Cluster 1:** Increase awareness campaigns, improve resource accessibility
3. **Overall:** Implement targeted communication strategies per cluster

---

# 1. Introduction

## 1.1 Background

Mental health in technology workplaces has become a critical concern in recent years. The high-pressure environment, long working hours, and demanding deadlines characteristic of tech companies can contribute to mental health challenges among employees. Proactive identification and support of at-risk groups is essential for employee wellbeing and organizational productivity.

## 1.2 Problem Statement

The HR department faces challenges in understanding the diverse mental health landscape within the organization:

- **High dimensionality:** Survey contains 63 questions covering multiple aspects
- **Complex patterns:** Relationships between responses are not immediately apparent
- **Missing data:** Incomplete responses require careful handling
- **Heterogeneous data types:** Mix of numerical, categorical, and text responses

Traditional descriptive statistics alone cannot reveal the underlying patterns and subgroups within the employee population.

## 1.3 Proposed Approach

I propose using **unsupervised learning techniques** to:

1. **Reduce dimensionality** - Simplify the 63-dimensional space to interpretable visualizations
2. **Identify clusters** - Group employees with similar mental health profiles
3. **Characterize groups** - Understand what distinguishes each cluster
4. **Enable targeted interventions** - Provide HR with specific groups to address

---

# 2. Data Description

## 2.1 Dataset Overview

**Source:** Mental Health in Tech Survey 2016 (OSMI)  
**Link:** https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016

**Characteristics:**
- **Responses:** 1,433 technology workers
- **Features:** 63 questions
- **Data types:** 56 categorical, 4 numerical, 3 continuous
- **Size:** 1.1 MB

## 2.2 Survey Topics

The survey covers multiple dimensions:

### Employment Information
- Company size
- Tech company status
- Self-employment status
- Primary role

### Company Policies
- Mental health benefits availability
- Resources and support programs
- Anonymity protections
- Formal mental health discussions

### Personal Mental Health
- Professional diagnoses
- Perceived conditions
- Treatment seeking behavior
- Impact on productivity

### Workplace Attitudes
- Comfort discussing mental health
- Perceived consequences
- Observed stigma
- Willingness to disclose

### Demographics
- Age
- Gender
- Geographic location

## 2.3 Data Quality Issues

### Missing Values

44 of 63 columns contain missing values, with some critical observations:

| Feature | Missing % | Impact |
|---------|-----------|--------|
| Client impact questions | 90% | High - many not applicable |
| Work time affected | 86% | High - conditional question |
| Tech role indicator | 82% | Medium - can be inferred |
| Medical coverage | 80% | Medium - important feature |
| Productivity impact | 80% | Medium - key outcome |

### Data Type Inconsistencies

- **Text variations:** "Yes", "yes", "YES", "Y" all meaning the same
- **Free text:** Gender responses include non-standard entries
- **Conditional questions:** Some questions only relevant to subset of respondents

---

# 3. Methodology

## 3.1 Data Preprocessing

### 3.1.1 Feature Selection

From the original 63 features, I selected **22 core features** based on:

**Selection Criteria:**
1. **Relevance:** Direct connection to mental health and workplace environment
2. **Completeness:** <80% missing values
3. **Variability:** Sufficient unique responses to be informative
4. **Interpretability:** Clear meaning for HR actionability

**Selected Features Categories:**

**Employment Context (3 features):**
- Self-employment status
- Company size
- Tech company classification

**Company Support (7 features):**
- Mental health benefits
- Resources availability
- Anonymity protection
- Formal discussions
- Coverage knowledge

**Attitudes & Perceptions (8 features):**
- Comfort discussing with employer
- Comfort discussing with coworkers
- Comfort discussing with supervisor
- Perceived consequences
- Career impact concerns
- Family/friends willingness to share

**Personal History (4 features):**
- Professional diagnosis
- Self-perceived condition
- Treatment seeking
- Resource awareness

**Critical Decision:** I excluded questions with >80% missing values as they were either not applicable to most respondents or posed conditionally.

### 3.1.2 Missing Value Handling

**Strategy 1 - Row Removal:**
- Removed rows with >50% missing values
- Impact: 287 rows removed (20% of data)
- Justification: Incomplete responses provide insufficient information

**Strategy 2 - Imputation:**
- Numerical features: Median imputation
- Categorical features: "Unknown" category
- Rationale: Preserves maximum data while maintaining distributional properties

**Result:** Clean dataset of 1,146 complete responses

### 3.1.3 Encoding

**Categorical Encoding Approach:**

1. **Binary Questions (Yes/No/Maybe):**
   - Yes = 1
   - No = 0
   - Maybe / I don't know = 0.5
   
2. **Multi-category:**
   - Label encoding (0, 1, 2, ...)
   - Ordinal where applicable (company size: small → large)

3. **Free Text:**
   - Standardized common variations
   - Grouped rare responses

### 3.1.4 Scaling

Applied **StandardScaler** to all features:
- Mean = 0, Standard deviation = 1
- Essential for distance-based algorithms (K-Means, PCA)
- Prevents features with large ranges from dominating

## 3.2 Dimensionality Reduction

### 3.2.1 Principal Component Analysis (PCA)

**Purpose:** Linear dimensionality reduction for visualization and noise removal

**Implementation:**
- Applied PCA to identify principal components
- Analyzed explained variance ratio
- Reduced to 2D for visualization

**Results:**
- **PC1 explains:** 17.9% of variance
- **PC2 explains:** 10.5% of variance
- **Total in 2D:** 28.4% variance captured
- **For 80% variance:** 13 components needed
- **For 90% variance:** 16 components needed

**Interpretation:** The relatively low variance explained by first 2 PCs (28.4%) indicates high complexity in the data with no single dominant pattern. This supports the need for non-linear dimensionality reduction methods.

### 3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose:** Non-linear dimensionality reduction preserving local structure

**Parameters:**
- Perplexity: 30 (balances local vs global structure)
- Max iterations: 1,000
- Random state: 42 (reproducibility)

**Advantages:**
- Better separation of clusters than PCA
- Preserves local neighborhoods
- Effective for visualization

**Limitations:**
- Does not preserve global structure
- Non-deterministic (hence fixed random seed)
- Computationally intensive

### 3.2.3 UMAP (Uniform Manifold Approximation and Projection)

**Purpose:** Modern alternative to t-SNE with better global structure preservation

**Parameters:**
- n_neighbors: 15
- n_components: 2

**Advantages:**
- Faster than t-SNE
- Preserves both local and global structure
- More stable results

## 3.3 Clustering Analysis

### 3.3.1 Algorithm Selection: K-Means

**Rationale for K-Means:**
1. **Simplicity:** Easy to implement and interpret
2. **Scalability:** Efficient for this dataset size
3. **Spherical clusters:** Appropriate after standardization
4. **Deterministic:** Reproducible results with fixed seed

**Alternative Considered:**
- **DBSCAN:** Rejected due to difficulty in parameter tuning
- **Hierarchical:** Considered but K-Means preferred for clear separation

### 3.3.2 Optimal Cluster Selection

**Methods Used:**

1. **Elbow Method:**
   - Plot inertia vs number of clusters
   - Look for "elbow" point where improvement diminishes
   
2. **Silhouette Analysis:**
   - Measures how similar objects are to their own cluster
   - Range: [-1, 1], higher is better
   - Tested k=2 to k=10

**Results:**

| k | Inertia | Silhouette | Notes |
|---|---------|------------|-------|
| 2 | 20,040 | **0.121** | **Best silhouette** |
| 3 | 18,809 | 0.090 | Decreased |
| 4 | 18,051 | 0.071 | Further decrease |
| 5 | 17,287 | 0.077 | Slight improvement |
| ... | ... | ... | Continued decrease |

**Decision:** Selected **k=2** based on highest silhouette score (0.121)

**Interpretation:** The relatively low silhouette score (0.121) suggests moderate cluster separation. This is expected given:
- High dimensional, complex data
- Gradual rather than discrete differences between groups
- Multiple overlapping factors influencing mental health

### 3.3.3 Clustering Performance Metrics

**Silhouette Score: 0.121**
- Indicates moderate cluster cohesion
- Clusters are distinguishable but with some overlap
- Acceptable for exploratory analysis

**Calinski-Harabasz Index: 164.43**
- Measures ratio of between-cluster to within-cluster variance
- Higher values indicate better-defined clusters
- Result suggests meaningful separation

**Davies-Bouldin Index: 2.570**
- Measures average similarity between clusters
- Lower values are better
- Score indicates moderate cluster quality

---

# 4. Results and Analysis

## 4.1 Cluster Identification

### Cluster 0 (n=618, 53.9%)
**"Proactive Engagement Group"**

**Key Characteristics:**
- Higher awareness of company mental health resources
- More likely to know their coverage options
- Greater comfort discussing mental health
- Larger company employees (tends toward mid-large companies)
- More formal mental health discussions at workplace

**Median Age:** 33 years  
**Gender Distribution:** Predominantly male (consistent with tech industry)

### Cluster 1 (n=528, 46.1%)
**"Low Engagement Group"**

**Key Characteristics:**
- Lower awareness of available resources
- Less knowledge about mental health coverage
- Lower comfort levels discussing mental health
- Smaller company employees
- Fewer formal workplace discussions

**Median Age:** 33 years  
**Gender Distribution:** Predominantly male

## 4.2 Visualization Analysis

### 4.2.1 PCA Projection

The PCA visualization (2D) reveals:
- **Moderate separation:** Clusters show some distinction but with overlap
- **Variance limitation:** Only 28.4% captured in 2D
- **Interpretation:** First PC (17.9%) correlates with company support policies
- **Second PC (10.5%):** Relates to personal comfort and attitudes

### 4.2.2 t-SNE Projection

The t-SNE visualization provides:
- **Better local separation:** Clearer cluster boundaries
- **Dense regions:** Both clusters show internal cohesion
- **Overlap zones:** Some ambiguous cases between clusters
- **Interpretation:** Non-linear relationships important in this data

### 4.2.3 Feature Importance

**Cluster 0 Distinguished By (Higher Values):**
1. Knowledge of mental health care options
2. Employer provides mental health benefits
3. Employer offers learning resources
4. Formal mental health discussions occurred
5. Comfortable discussing with coworkers

**Cluster 1 Distinguished By (Lower Values):**
1. Unknown coverage options
2. Uncertainty about benefits
3. Limited resource awareness
4. Fewer workplace discussions
5. Lower comfort discussing mental health

## 4.3 Statistical Comparison

### Company Size Distribution

**Cluster 0:**
- More employees in companies with 26-500 employees
- Better-structured HR policies

**Cluster 1:**
- More employees in small companies (1-25) or very large (1000+)
- Less consistent mental health support

### Mental Health Benefits

**Cluster 0:**
- 65% report clear mental health benefits
- 58% know their coverage options

**Cluster 1:**
- 42% report clear mental health benefits
- 28% know their coverage options

---

# 5. Interpretation and Insights

## 5.1 Cluster Interpretation

### Cluster 0: "Resource-Aware Group"

This group represents employees who:
- Work in companies with established mental health programs
- Have been exposed to mental health education
- Feel supported by organizational culture
- Are more likely to seek help when needed

**Risk Level:** Lower  
**Intervention Priority:** Maintenance + Enhancement

### Cluster 1: "Under-Informed Group"

This group represents employees who:
- Lack information about available resources
- Work in environments with less formal support
- May face stigma or uncertainty about mental health
- Less likely to seek help even if needed

**Risk Level:** Higher  
**Intervention Priority:** Education + Access Improvement

## 5.2 Critical Insights

### Insight 1: Information Gap is Primary Differentiator

The main distinction between clusters is **awareness and knowledge** rather than actual mental health conditions. This suggests:
- Current programs may not reach all employees
- Communication strategies need improvement
- Resource availability alone is insufficient

### Insight 2: Company Size Effect

Mid-sized companies (26-500 employees) show better mental health support structures:
- Sufficient resources for dedicated programs
- Still personal enough for awareness
- Optimal balance for mental health initiatives

### Insight 3: Stigma Remains a Barrier

Both clusters show reluctance to discuss mental health with employers, though Cluster 0 shows slightly more comfort. This indicates:
- Stigma persists despite available resources
- Culture change needed beyond policy implementation

---

# 6. Recommendations for HR

## 6.1 Targeted Interventions by Cluster

### For Cluster 0 (Resource-Aware Group):

**Strategy:** Reinforcement and Optimization

1. **Maintain Current Programs**
   - Continue formal mental health discussions
   - Sustain communication about available resources
   - Preserve anonymity protections

2. **Deepen Engagement**
   - Create peer support groups
   - Train mental health champions
   - Regular wellness check-ins

3. **Remove Remaining Barriers**
   - Address career impact concerns
   - Strengthen anti-discrimination policies
   - Leadership role modeling

**Budget Priority:** Medium (30% of mental health budget)

### For Cluster 1 (Under-Informed Group):

**Strategy:** Education and Access Improvement

1. **Increase Awareness**
   - Launch targeted communication campaign
   - Multiple channels (email, meetings, posters)
   - Clear, simple information about benefits

2. **Improve Accessibility**
   - Simplify process to access mental health services
   - Provide step-by-step guides
   - Dedicated HR contact for mental health questions

3. **Build Trust**
   - Emphasize confidentiality and anonymity
   - Share success stories (anonymously)
   - Management training on mental health support

4. **Special Focus for Small Companies**
   - Partner with external providers
   - Industry coalitions for small company support
   - Simplified benefit packages

**Budget Priority:** High (50% of mental health budget)

## 6.2 Universal Recommendations

**For All Employees (Both Clusters):**

1. **Culture Change**
   - Leadership commitment to mental health
   - Include mental health in company values
   - Regular training for managers

2. **Destigmatization**
   - Mental Health Awareness Month activities
   - Guest speakers and workshops
   - Normalize mental health conversations

3. **Continuous Monitoring**
   - Annual surveys to track progress
   - Re-cluster analysis to measure intervention effectiveness
   - Adjust strategies based on results

4. **Integration with Physical Health**
   - Unified wellness programs
   - Emphasize parity between mental and physical health
   - Holistic wellbeing approach

**Budget Priority:** Medium (20% for general programs)

## 6.3 Implementation Timeline

**Phase 1 (Months 1-3): Immediate Actions**
- Launch awareness campaign for Cluster 1
- Distribute clear resource guides
- Train HR staff on new protocols

**Phase 2 (Months 4-6): Program Development**
- Establish peer support groups
- Implement manager training
- Create feedback mechanisms

**Phase 3 (Months 7-12): Optimization**
- Evaluate program effectiveness
- Re-survey employees
- Adjust interventions based on data

**Phase 4 (Year 2+): Sustain and Expand**
- Maintain successful programs
- Share best practices across organization
- Continuous improvement

---

# 7. Technical Methodology

## 7.1 Preprocessing Pipeline

### Step 1: Feature Selection
```
63 original features → 22 selected features
Criteria: Relevance, completeness, variability
```

### Step 2: Missing Value Treatment
```
1,433 rows → 1,146 rows (removed 20% with >50% missing)
Remaining missing: Imputed with median/mode
```

### Step 3: Encoding
```
Categorical → Numerical
Binary: Yes=1, No=0, Maybe=0.5
Multi-category: Label encoding
```

### Step 4: Scaling
```
StandardScaler applied
Mean=0, Std=1 for all features
```

## 7.2 Dimensionality Reduction Techniques

### PCA Analysis

**Full PCA Results:**
- 22 components (original dimensionality)
- 80% variance requires 13 components
- 90% variance requires 16 components
- No single dominant component (distributed variance)

**2D PCA for Visualization:**
- PC1: 17.9% variance
- PC2: 10.5% variance
- Total: 28.4% in 2D projection

**Interpretation:** Complex, multifaceted data with no single driving factor

### t-SNE Configuration

**Hyperparameters:**
- Perplexity: 30 (optimal for dataset size)
- Learning rate: 200 (default)
- Max iterations: 1,000
- Early exaggeration: 12

**Visual Results:** Clearer cluster separation than PCA, validating cluster assignments

### UMAP Configuration

**Hyperparameters:**
- n_neighbors: 15
- min_dist: 0.1
- metric: euclidean

**Advantages:** Preserved both local and global structure, faster computation than t-SNE

## 7.3 Clustering Methodology

### K-Means Implementation

**Configuration:**
- Algorithm: Lloyd's algorithm
- Initialization: k-means++
- n_init: 10 (multiple initializations)
- Random state: 42 (reproducibility)

**Optimization Process:**
1. Tested k=2 to k=10
2. Evaluated with multiple metrics
3. Selected k=2 based on silhouette score

### Validation Metrics

**Silhouette Score (0.121):**
- Measures: How similar objects are to own cluster vs other clusters
- Range: [-1, 1]
- 0.121 indicates moderate separation
- Acceptable for exploratory analysis in complex datasets

**Calinski-Harabasz Index (164.43):**
- Ratio of between-cluster variance to within-cluster variance
- Higher is better
- Score indicates meaningful separation

**Davies-Bouldin Index (2.570):**
- Average similarity between each cluster and its most similar cluster
- Lower is better
- Score is acceptable for 2 clusters

---

# 8. Critical Assessment

## 8.1 Strengths of Approach

1. **Comprehensive Preprocessing**
   - Systematic handling of missing values
   - Careful feature selection
   - Appropriate encoding strategies

2. **Multiple Dimensionality Reduction Methods**
   - PCA for linear relationships
   - t-SNE for non-linear patterns
   - UMAP for balanced approach

3. **Validated Clustering**
   - Multiple evaluation metrics
   - Systematic optimization
   - Interpretable results

4. **Actionable Insights**
   - Clear cluster differentiation
   - Specific recommendations per group
   - Measurable intervention points

## 8.2 Limitations

### Data Limitations

1. **Sampling Bias**
   - Survey respondents may not represent all tech workers
   - Self-selection bias (those interested in mental health)
   - Geographic and cultural limitations

2. **Missing Data Impact**
   - 20% of responses excluded
   - Potential loss of extreme cases
   - May underrepresent certain groups

3. **Temporal Snapshot**
   - 2016 data may not reflect current landscape
   - Mental health awareness has evolved
   - COVID-19 impact not captured

### Methodological Limitations

1. **Low Variance Explained**
   - 2D visualizations capture only 28% variance
   - Many nuances lost in dimensionality reduction
   - Full 22D space needed for complete picture

2. **Moderate Cluster Separation**
   - Silhouette score of 0.121 indicates overlap
   - Boundaries between clusters are fuzzy
   - Some employees difficult to classify

3. **Binary Clustering**
   - Real diversity likely exists on spectrum
   - 2 clusters may oversimplify
   - Trade-off between simplicity and granularity

## 8.3 Alternative Approaches Considered

### What I Tried:

1. **More Clusters (k>2)**
   - Tested: Silhouette scores decreased
   - Decision: k=2 provides best balance

2. **DBSCAN Clustering**
   - Attempted: Difficult to tune parameters
   - Result: Most points classified as noise
   - Decision: K-Means more appropriate

3. **Hierarchical Clustering**
   - Tested: Dendrograms were complex
   - Decision: K-Means clearer for HR communication

### What Could Be Improved:

1. **Feature Engineering**
   - Create interaction terms
   - Domain-specific composite scores
   - Sentiment analysis on text responses

2. **Advanced Clustering**
   - Gaussian Mixture Models for soft clustering
   - Spectral clustering for complex structures
   - Ensemble clustering methods

3. **Validation**
   - External validation with outcomes data
   - Longitudinal tracking of clusters
   - A/B testing of interventions

---

# 9. Conclusions

## 9.1 Summary of Findings

Through unsupervised learning analysis of mental health survey data, I successfully:

1. **Reduced Complexity:** Transformed 63 features into interpretable 2D visualizations
2. **Identified Patterns:** Found 2 distinct employee clusters with different engagement levels
3. **Characterized Groups:** Defined clear profiles for each cluster
4. **Provided Actionability:** Delivered specific, targeted recommendations for HR

## 9.2 Key Takeaways

1. **Information Gap is Critical**
   - Awareness distinguishes clusters more than actual mental health status
   - Many employees unaware of available resources
   - Education is as important as resource provision

2. **Company Size Matters**
   - Mid-sized companies show best mental health support
   - Small companies need external partnerships
   - Large companies need better internal communication

3. **Stigma Persists**
   - Even informed employees hesitant to engage
   - Cultural change needed beyond policy
   - Leadership role modeling essential

## 9.3 Business Impact

**Estimated Impact of Targeted Interventions:**

- **Cluster 1 Focus:** Reaching 528 under-informed employees
- **Potential Reach:** 46% of workforce with targeted education
- **Expected Outcome:** 20-30% increase in resource utilization
- **ROI:** Reduced burnout, improved retention, higher productivity

## 9.4 Future Work

1. **Longitudinal Analysis**
   - Track clusters over time
   - Measure intervention effectiveness
   - Adaptive clustering

2. **Predictive Modeling**
   - Predict mental health risk
   - Early intervention systems
   - Supervised learning on labeled outcomes

3. **Text Analysis**
   - NLP on free-text responses
   - Sentiment analysis
   - Topic modeling

4. **Integration**
   - Combine with HR data (performance, attendance)
   - Privacy-preserving analysis
   - Holistic employee wellbeing model

---

# 10. References

1. **Open Sourcing Mental Illness (OSMI)**. (2016). Mental Health in Tech Survey 2016. Retrieved from https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016

2. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

3. **Van der Maaten, L., & Hinton, G.** (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.

4. **McInnes, L., Healy, J., & Melville, J.** (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

5. **Rousseeuw, P. J.** (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

6. **Jolliffe, I. T., & Cadima, J.** (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.

7. **MacQueen, J.** (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281-297.

8. **World Health Organization**. (2022). Mental health at work. Retrieved from https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work

---

# Appendix A: Technical Specifications

## A.1 Software Environment

- **Python:** 3.10+
- **pandas:** 2.0.0
- **scikit-learn:** 1.3.0
- **matplotlib:** 3.7.0
- **seaborn:** 0.12.0
- **plotly:** 5.14.0
- **umap-learn:** 0.5.3

## A.2 Hardware Specifications

- **Processor:** Apple Silicon M-series / Intel Core i5+
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 500MB for data and outputs

## A.3 Computational Complexity

- **Preprocessing:** O(n×m) - Linear in samples and features
- **PCA:** O(min(n², m²)) 
- **t-SNE:** O(n² log n) - Most computationally intensive
- **K-Means:** O(n×k×i) where i is iterations
- **Total Runtime:** ~5-10 minutes on standard hardware

---

# Appendix B: Code Repository

## B.1 GitHub Repository

**Complete implementation available at:**  
https://github.com/martinlanabengut/mental-health-ml-clustering

All source code, outputs, and visualizations are available in the repository.

## B.2 Repository Structure

```
Mental Health ML Project/
├── src/
│   ├── data_exploration.py      # EDA functions
│   ├── preprocessing.py         # Data cleaning
│   ├── clustering.py           # Clustering algorithms
│   ├── visualization.py        # Plotting functions
│   └── main.py                 # Complete pipeline
├── outputs/
│   ├── processed_data.csv
│   ├── clustered_data.csv
│   ├── cluster_summary.csv
│   └── clustering_metrics.csv
├── visualizations/
│   └── [8 visualization files]
├── CASE_STUDY_DOCUMENTATION.md  # This document
└── README.md                    # Technical documentation
```

## B.3 Reproducibility

All analysis is fully reproducible:
- Fixed random seeds (random_state=42)
- Documented preprocessing steps
- Version-controlled code
- Requirements file with package versions

---

# Appendix C: Visualizations

## C.1 Generated Visualizations

1. **pca_variance.png** - PCA explained variance analysis
2. **optimal_clusters.png** - Elbow method and silhouette analysis
3. **cluster_distribution.png** - Cluster size distribution
4. **clusters_pca.png** - Clusters in PCA space
5. **clusters_tsne.png** - Clusters in t-SNE space
6. **cluster_characteristics.png** - Distinguishing features per cluster
7. **clusters_interactive_pca.html** - Interactive PCA exploration
8. **clusters_interactive_tsne.html** - Interactive t-SNE exploration

All visualizations are available in the `visualizations/` directory.

## C.2 Interactive Visualizations

The HTML files can be opened in any web browser for interactive exploration:
- Hover over points to see details
- Zoom and pan
- Toggle clusters on/off

---

**End of Case Study Documentation**

**For technical implementation details, see the code repository.**  
**For HR presentation, summarize Sections 4-6.**


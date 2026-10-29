# Customer Segmentation with K-means

Unsupervised learning project that groups mall customers into meaningful segments using **K-means implemented from scratch in NumPy**, then validates it against scikit-learn.

> Built as a practical application of **Module 1 (Clustering & Anomaly Detection)** of the Unsupervised Learning course, not of the whole course.

## Problem
A mall has data on its customers but no labels telling which customer is "VIP" or "budget". The goal is to discover natural customer groups so each can be targeted with a different marketing strategy.

## Dataset
[Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle), 200 customers.

| Feature | Description |
|---|---|
| Age | Customer age |
| Annual Income (k$) | Yearly income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall based on spending behavior |

Download `Mall_Customers.csv` and place it next to the script.

## Approach
1. **EDA:** inspect distributions and check for missing values.
2. **Scaling:** `StandardScaler`, because K-means is distance-based.
3. **K-means from scratch:**
   - k-means++ initialization
   - assignment / centroid-update loop until convergence
   - multiple random restarts (`n_init=10`), keeping the lowest cost
   - empty-cluster handling
4. **Choosing K:** Elbow method (inertia) together with Silhouette score.
5. **Validation:** compare with `sklearn.cluster.KMeans` (inertia and Adjusted Rand Index).
6. **Interpretation:** profile each cluster (mean Age / Income / Spending) and give it a business meaning.
7. **Visualization:** Income vs Spending scatter with centroids, plus a 2D PCA view.

## Project structure
```
customer-segmentation/
├── customer_segmentation.py
├── Mall_Customers.csv            # download from Kaggle
├── requirements.txt
└── outputs_segmentation/         # generated
    ├── choose_k.png
    ├── clusters.png
    ├── cluster_profiles.csv
    └── customers_with_segments.csv
```

## How to run
```bash
pip install -r requirements.txt
python customer_segmentation.py          # uses Mall_Customers.csv
python customer_segmentation.py --demo   # synthetic data, to test the pipeline
```

## Results
<!-- Fill this in after running on the real dataset -->
- Best K (silhouette): `__`
- Silhouette score: `__`
- Scratch vs sklearn inertia: `__` vs `__`

| Cluster | Avg Age | Avg Income | Avg Spending | Size | Segment |
|---|---|---|---|---|---|
| 0 | | | | | |

![Choosing K](outputs_segmentation/choose_k.png)
![Clusters](outputs_segmentation/clusters.png)

## Key takeaways
- Scaling matters: without it, income dominates the distance.
- Elbow and Silhouette should be read together; neither is conclusive alone.
- K-means assumes roughly spherical, similar-sized clusters, which is a limitation.

## Possible improvements
- Compare with DBSCAN and Gaussian Mixture Models
- Add Gender as a one-hot feature
- Build a small Streamlit dashboard

## Tech stack
Python, NumPy, pandas, Matplotlib, scikit-learn

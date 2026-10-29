
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

OUT = "outputs_segmentation"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------
# 1) K-means from scratch (NumPy) with k-means++ initialization
# ---------------------------------------------------------------
class KMeansScratch:
    def __init__(self, k, max_iters=300, n_init=10, tol=1e-6, seed=42):
        self.k, self.max_iters, self.n_init = k, max_iters, n_init
        self.tol, self.seed = tol, seed
        self.centroids, self.labels_, self.inertia_ = None, None, None

    def _init_centroids(self, X, rng):
        # k-means++: pick far-apart starting centroids
        n = X.shape[0]
        C = [X[rng.integers(n)]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(2), axis=1)
            probs = d2 / d2.sum()
            C.append(X[rng.choice(n, p=probs)])
        return np.array(C)

    @staticmethod
    def _assign(X, C):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        return d2.argmin(1), d2.min(1).sum()

    def _fit_once(self, X, rng):
        C = self._init_centroids(X, rng)
        for _ in range(self.max_iters):
            labels, inertia = self._assign(X, C)
            newC = C.copy()
            for j in range(self.k):
                pts = X[labels == j]
                if len(pts):
                    newC[j] = pts.mean(0)
                else:  # empty cluster -> re-seed randomly
                    newC[j] = X[rng.integers(len(X))]
            shift = np.abs(newC - C).max()
            C = newC
            if shift < self.tol:
                break
        labels, inertia = self._assign(X, C)
        return C, labels, inertia

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):  # run several times, keep lowest cost
            res = self._fit_once(X, rng)
            if best is None or res[2] < best[2]:
                best = res
        self.centroids, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        return self._assign(X, self.centroids)[0]


# ---------------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------------
def load_data(demo=False, path="Mall_Customers.csv"):
    if demo or not os.path.exists(path):
        print(">> Using synthetic demo data (Mall_Customers.csv not found)")
        rng = np.random.default_rng(0)
        centers = [(25, 25, 78), (45, 25, 20), (32, 85, 82), (42, 88, 17), (45, 55, 50)]
        rows = []
        for a, inc, sp in centers:
            rows.append(np.c_[rng.normal(a, 5, 40), rng.normal(inc, 6, 40), rng.normal(sp, 8, 40)])
        X = np.vstack(rows)
        df = pd.DataFrame(X, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        df["Gender"] = rng.choice(["Male", "Female"], len(df))
        return df
    return pd.read_csv(path)


def main(demo):
    df = load_data(demo)
    print(df.head(), "\n")
    print(df.describe(), "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X_raw = df[features].values
    X = StandardScaler().fit_transform(X_raw)  # K-means is distance-based -> scale!

    # ------------------ choose K: Elbow + Silhouette ------------------
    ks = range(2, 11)
    inertias, sils = [], []
    for k in ks:
        km = KMeansScratch(k).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))
    best_k = list(ks)[int(np.argmax(sils))]
    print(f"Silhouette scores: { {k: round(s, 3) for k, s in zip(ks, sils)} }")
    print(f"Best K by silhouette = {best_k}\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(ks), inertias, "o-"); ax[0].set_title("Elbow method")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia (cost J)")
    ax[1].plot(list(ks), sils, "o-", color="green"); ax[1].set_title("Silhouette score")
    ax[1].set_xlabel("K"); ax[1].set_ylabel("Score")
    plt.tight_layout(); plt.savefig(f"{OUT}/choose_k.png", dpi=150); plt.close()

    # ------------------ final model + compare with sklearn ------------------
    K = best_k  # you can override manually after looking at the elbow plot
    km = KMeansScratch(K, n_init=10).fit(X)
    sk = SKKMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
    print(f"Scratch inertia: {km.inertia_:.2f} | sklearn inertia: {sk.inertia_:.2f}")
    print(f"Agreement (Adjusted Rand Index): {adjusted_rand_score(km.labels_, sk.labels_):.3f}\n")

    df["Cluster"] = km.labels_

    # ------------------ profile each cluster ------------------
    profile = df.groupby("Cluster")[features].mean().round(1)
    profile["Size"] = df["Cluster"].value_counts().sort_index()
    print("Cluster profiles:\n", profile, "\n")
    profile.to_csv(f"{OUT}/cluster_profiles.csv")

    # Simple automatic naming based on income/spending (relative to the data mean)
    inc_m, sp_m = df[features[1]].mean(), df[features[2]].mean()
    names = {}
    for c, r in profile.iterrows():
        hi_inc, hi_sp = r[features[1]] > inc_m, r[features[2]] > sp_m
        names[c] = {(True, True): "High income - High spending (VIP)",
                    (True, False): "High income - Low spending (Careful)",
                    (False, True): "Low income - High spending (Impulsive)",
                    (False, False): "Low income - Low spending (Budget)"}[(hi_inc, hi_sp)]
    df["Segment"] = df["Cluster"].map(names)
    print(df["Segment"].value_counts(), "\n")

    # ------------------ visualizations ------------------
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    sc = ax1.scatter(df[features[1]], df[features[2]], c=df["Cluster"], cmap="tab10", s=35)
    cents = StandardScaler().fit(X_raw).inverse_transform(km.centroids)
    ax1.scatter(cents[:, 1], cents[:, 2], c="black", marker="X", s=200, label="Centroids")
    ax1.set_xlabel(features[1]); ax1.set_ylabel(features[2])
    ax1.set_title("Income vs Spending"); ax1.legend()

    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(Z[:, 0], Z[:, 1], c=df["Cluster"], cmap="tab10", s=35)
    ax2.set_title(f"PCA view (explained var = {pca.explained_variance_ratio_.sum():.0%})")
    plt.tight_layout(); plt.savefig(f"{OUT}/clusters.png", dpi=150); plt.close()

    df.to_csv(f"{OUT}/customers_with_segments.csv", index=False)
    print(f"Done. Results saved in ./{OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args, _ = ap.parse_known_args()
    main(args.demo)

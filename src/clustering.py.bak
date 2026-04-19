# ==========================================
# RETAIL ML PROJECT - CLUSTERING SCRIPT
# ==========================================
# Unsupervised learning — K-Means Clustering
# Goal: Discover natural customer segments
# without using the Churn label.
#
# Steps:
#   1. Load preprocessed data
#   2. Elbow method → find optimal K
#   3. Silhouette score → confirm K
#   4. Train K-Means
#   5. Visualize clusters (PCA 2D)
#   6. Profile each cluster
#   7. Save results
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ==========================================
# 0️⃣ SETUP
# ==========================================

os.makedirs("reports",          exist_ok=True)
os.makedirs("models",           exist_ok=True)
os.makedirs("data/processed",   exist_ok=True)

RANDOM_STATE = 42
PALETTE      = ["#2c5f8a", "#e67e22", "#27ae60", "#c0392b", "#8e44ad", "#16a085"]


# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

print("Loading preprocessed data...")

X_train = pd.read_csv("data/train_test/X_train.csv")
X_test  = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

# Combine train + test for clustering
# (clustering is unsupervised — we use all data)
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)

print(f"  Total customers for clustering : {len(X_all)}")

# Use only numeric columns
numeric_cols = X_all.select_dtypes(include=["int64", "float64"]).columns
X_num        = X_all[numeric_cols].fillna(0)

print(f"  Numeric features used          : {X_num.shape[1]}")


# ==========================================
# 2️⃣ ELBOW METHOD — Find optimal K
# ==========================================
# The elbow method runs K-Means for K=2..10
# and plots the inertia (sum of squared distances
# from each point to its cluster center).
# The "elbow" point = optimal K.
# ==========================================

print("\nRunning Elbow Method (K = 2 to 10)...")

k_range  = range(2, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_num)
    inertias.append(km.inertia_)
    print(f"  K={k}  Inertia={km.inertia_:,.0f}")

# --- Plot elbow curve ---
plt.figure(figsize=(9, 5))
plt.plot(k_range, inertias, marker="o", color="#2c5f8a",
         linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)

# Highlight elbow (simple heuristic: biggest drop)
deltas       = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
elbow_k      = list(k_range)[deltas.index(max(deltas)) + 1]
elbow_inertia = inertias[list(k_range).index(elbow_k)]

plt.axvline(x=elbow_k, color="#c0392b", linestyle="--", linewidth=1.5,
            label=f"Suggested elbow: K={elbow_k}")
plt.scatter([elbow_k], [elbow_inertia], color="#c0392b", zorder=5, s=100)

plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Inertia (Within-cluster SSE)", fontsize=11)
plt.title("Elbow Method — Optimal Number of Clusters", fontsize=13)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/elbow_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✅ Elbow curve saved → reports/elbow_curve.png")
print(f"  📌 Suggested K from elbow : {elbow_k}")


# ==========================================
# 3️⃣ SILHOUETTE SCORE — Confirm K
# ==========================================
# Silhouette score measures how well each point
# fits its cluster vs neighbouring clusters.
# Score range: -1 (bad) → 0 → 1 (perfect)
# Higher = better separated clusters.
# ==========================================

print("\nComputing Silhouette Scores...")

sil_scores = {}

for k in range(2, 8):
    km    = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_num)
    score  = silhouette_score(X_num, labels, sample_size=1000, random_state=RANDOM_STATE)
    sil_scores[k] = score
    print(f"  K={k}  Silhouette={score:.4f}")

best_k_sil = max(sil_scores, key=sil_scores.get)
print(f"\n  📌 Best K from silhouette : {best_k_sil} (score={sil_scores[best_k_sil]:.4f})")

# --- Plot silhouette scores ---
plt.figure(figsize=(8, 4))
ks     = list(sil_scores.keys())
scores = list(sil_scores.values())
bars   = plt.bar(ks, scores,
                 color=["#2c5f8a" if k != best_k_sil else "#c0392b" for k in ks],
                 edgecolor="white")
for bar, val in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.003,
             f"{val:.3f}", ha="center", fontsize=9)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores per K")
plt.tight_layout()
plt.savefig("reports/silhouette_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Silhouette scores saved → reports/silhouette_scores.png")

# --- Choose final K ---
# Use silhouette best K (more reliable than elbow heuristic)
FINAL_K = best_k_sil
print(f"\n  🎯 Final K selected : {FINAL_K}")


# ==========================================
# 4️⃣ TRAIN FINAL K-MEANS
# ==========================================

print(f"\nTraining K-Means with K={FINAL_K}...")

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)

cluster_labels = kmeans.fit_predict(X_num)

print(f"  ✅ K-Means trained")
print(f"  Cluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f"    Cluster {c} : {n} customers ({n/len(cluster_labels)*100:.1f}%)")

# Save model
joblib.dump(kmeans, "models/kmeans_model.pkl")
print("  ✅ models/kmeans_model.pkl saved")


# ==========================================
# 5️⃣ PCA 2D VISUALIZATION
# ==========================================
# Reduce to 2 dimensions with PCA to visualize
# the clusters on a 2D scatter plot.
# ==========================================

print("\nVisualizing clusters with PCA 2D...")

pca_2d   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X_num)

var_pc1  = pca_2d.explained_variance_ratio_[0] * 100
var_pc2  = pca_2d.explained_variance_ratio_[1] * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: colored by Cluster ---
ax = axes[0]
for i in range(FINAL_K):
    mask = cluster_labels == i
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               c=PALETTE[i % len(PALETTE)],
               label=f"Cluster {i}",
               alpha=0.45, s=15, edgecolors="none")

ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=10)
ax.set_title("K-Means Clusters (PCA 2D)", fontsize=12)
ax.legend(markerscale=2, fontsize=9)

# --- Right: colored by Churn ---
ax2 = axes[1]
churn_colors = {0: "#2c5f8a", 1: "#c0392b"}
churn_labels = {0: "Fidèle (0)", 1: "Churn (1)"}
for label, color in churn_colors.items():
    mask = y_all.values == label
    ax2.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=color, label=churn_labels[label],
                alpha=0.35, s=15, edgecolors="none")

ax2.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=10)
ax2.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=10)
ax2.set_title("Actual Churn Labels (PCA 2D)", fontsize=12)
ax2.legend(markerscale=2, fontsize=9)

plt.suptitle("PCA 2D — Clusters vs Actual Churn", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("reports/clusters_pca2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/clusters_pca2d.png saved")


# ==========================================
# 6️⃣ CLUSTER PROFILING
# ==========================================
# Attach cluster labels to original data and
# compute the mean of key features per cluster
# to understand what each cluster represents.
# ==========================================

print("\nProfiling clusters...")

# Reload raw-ish numeric features for readable profiling
X_profile = X_all[numeric_cols].copy()
X_profile["Cluster"] = cluster_labels
X_profile["Churn"]   = y_all.values

# Key features to profile
profile_features = [
    "Recency", "Frequency", "MonetaryTotal",
    "CustomerTenureDays", "Satisfaction",
    "ReturnRatio", "CancelledTransactions",
    "UniqueProducts", "SupportTickets",
    "Churn"
]

# Keep only features that exist
profile_features = [f for f in profile_features if f in X_profile.columns]

profile = X_profile.groupby("Cluster")[profile_features].mean().round(3)
print("\n  Cluster Profiles (mean values):")
print(profile.to_string())

# --- Heatmap of cluster profiles ---
profile_plot = profile.drop(columns=["Churn"], errors="ignore")

# Normalize each column for heatmap readability
profile_norm = (profile_plot - profile_plot.min()) / (profile_plot.max() - profile_plot.min() + 1e-9)

plt.figure(figsize=(12, max(4, FINAL_K + 1)))
sns.heatmap(
    profile_norm,
    annot=profile_plot.round(2),
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Normalized Value"},
    yticklabels=[f"Cluster {i}" for i in range(FINAL_K)]
)
plt.title("Cluster Profiles — Mean Feature Values", fontsize=13)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("reports/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/cluster_profiles.png saved")

# --- Churn rate per cluster bar chart ---
churn_per_cluster = X_profile.groupby("Cluster")["Churn"].mean() * 100

plt.figure(figsize=(8, 4))
bars = plt.bar(
    [f"Cluster {i}" for i in churn_per_cluster.index],
    churn_per_cluster.values,
    color=PALETTE[:FINAL_K],
    edgecolor="white"
)
for bar, val in zip(bars, churn_per_cluster.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.axhline(y=y_all.mean()*100, color="#888", linestyle="--",
            linewidth=1.2, label=f"Overall churn rate ({y_all.mean()*100:.1f}%)")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate per Cluster", fontsize=13)
plt.legend()
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("reports/cluster_churn_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/cluster_churn_rate.png saved")


# ==========================================
# 7️⃣ CLUSTER NAMING (Business Interpretation)
# ==========================================
# Based on the profiles, give each cluster
# a business-friendly name.
# ==========================================

print("\nInterpreting clusters...")

# Auto-assign names based on Churn rate and Recency
cluster_names = {}
for i in range(FINAL_K):
    churn_rate = churn_per_cluster[i]
    recency    = profile.loc[i, "Recency"]   if "Recency"    in profile.columns else 0
    frequency  = profile.loc[i, "Frequency"] if "Frequency"  in profile.columns else 0

    if churn_rate > 60:
        name = f"Cluster {i} — 🔴 Lost / Churned"
    elif churn_rate > 35:
        name = f"Cluster {i} — 🟠 At Risk"
    elif recency < 30 and frequency > 0:
        name = f"Cluster {i} — 🟢 Active Loyal"
    else:
        name = f"Cluster {i} — 🔵 Occasional"

    cluster_names[i] = name
    print(f"  {name}  |  Churn rate: {churn_rate:.1f}%")


# ==========================================
# 8️⃣ SAVE RESULTS
# ==========================================

print("\nSaving cluster assignments...")

X_all_with_clusters              = X_all.copy()
X_all_with_clusters["Cluster"]   = cluster_labels
X_all_with_clusters["Churn"]     = y_all.values

X_all_with_clusters.to_csv(
    "data/processed/customers_with_clusters.csv",
    index=False
)
print("  ✅ data/processed/customers_with_clusters.csv saved")

# Save profile summary
profile.to_csv("reports/cluster_profiles_summary.csv")
print("  ✅ reports/cluster_profiles_summary.csv saved")


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*55)
print("  CLUSTERING COMPLETE")
print("="*55)
print(f"  Algorithm   : K-Means")
print(f"  K selected  : {FINAL_K} clusters")
print(f"  Dataset     : {len(X_all)} customers")
print(f"\n  Reports generated:")
print("    reports/elbow_curve.png")
print("    reports/silhouette_scores.png")
print("    reports/clusters_pca2d.png")
print("    reports/cluster_profiles.png")
print("    reports/cluster_churn_rate.png")
print("    reports/cluster_profiles_summary.csv")
print("    data/processed/customers_with_clusters.csv")
print("="*55)
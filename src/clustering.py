# ==========================================
# RETAIL ML PROJECT - CLUSTERING SCRIPT
# ==========================================
# Unsupervised learning — K-Means Clustering
# Goal: Discover natural customer segments
# without using the Churn label.
#
# FIXES vs original:
#   ✅ Removed "Recency" from profile features — dropped as leaky
#   ✅ Fixed column names: SatisfactionScore, SupportTicketsCount
#   ✅ Fixed elbow heuristic off-by-one error
#   ✅ Profiling uses inverse-scaled values (readable business numbers)
#   ✅ Removed "Churn" duplicate from profile_features list
#   ✅ Cluster auto-naming no longer references Recency
#   ✅ y_all index alignment fixed with reset_index
#   ✅ Profile features guarded with existence check
#
# Steps:
#   1. Load preprocessed data
#   2. Elbow method → find optimal K
#   3. Silhouette score → confirm K
#   4. Train K-Means
#   5. Visualize clusters (PCA 2D)
#   6. Profile each cluster (inverse-scaled values)
#   7. Save results
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score


# ==========================================
# 0️⃣ SETUP
# ==========================================

os.makedirs("reports",        exist_ok=True)
os.makedirs("models",         exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

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

# Load scaler for inverse-transform (readable profiling)
scaler = joblib.load("models/scaler.pkl")

# Combine train + test — clustering is unsupervised, use all data
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)   # ✅ reset_index via ignore_index

print(f"  Total customers for clustering : {len(X_all)}")

# Use only numeric columns (already scaled by StandardScaler)
# AFTER
CLUSTERING_FEATURES = [
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "ReturnRatio",
    "CancelledTransactions",
    "UniqueProducts",
    "SupportTicketsCount",
    "EngagementScore",
    "DisengagementScore",
    "AvgBasketValue",
    "Country_TargetEnc",
]
clustering_cols = [c for c in CLUSTERING_FEATURES if c in X_all.columns]
X_num           = X_all[clustering_cols].fillna(0)

# Re-scale ONLY these columns (they're already StandardScaler-transformed,
# but now we're subsetting so the relative scale is preserved — no re-scaling needed)
print(f"  Clustering on {len(clustering_cols)} interpretable features: {clustering_cols}")

print(f"  Numeric features used          : {X_num.shape[1]}")
print(f"  Note: features are StandardScaler-transformed (mean=0, std=1)")

# Remove outliers before clustering.
# Customers with any feature beyond 4 standard deviations are data anomalies
# that K-Means will isolate into their own tiny clusters (as seen with
# Cluster 2: 13 customers with CancelledTransactions=97, UniqueProducts=613,
# and Cluster 3: 53 customers with negative MonetaryTotal in original scale).
# These clusters have no business meaning and distort the main segmentation.
z_scores      = np.abs(X_num)                          # already StandardScaler-transformed → z-scores
outlier_mask  = (z_scores > 4).any(axis=1)
n_outliers    = outlier_mask.sum()
print(f"  Outliers removed (|z| > 4)     : {n_outliers} customers")

X_num  = X_num[~outlier_mask].reset_index(drop=True)
y_all  = y_all[~outlier_mask].reset_index(drop=True)
X_all  = X_all[~outlier_mask].reset_index(drop=True)
print(f"  Customers after outlier removal: {len(X_num)}")
# ==========================================
# 2️⃣ ELBOW METHOD — Find optimal K
# ==========================================
# Runs K-Means for K=2..10 and plots inertia
# (sum of squared distances to cluster center).
# The "elbow" = point of diminishing returns.
# ==========================================

print("\nRunning Elbow Method (K = 2 to 10)...")

k_range  = range(2, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_num)
    inertias.append(km.inertia_)
    print(f"  K={k}  Inertia={km.inertia_:,.0f}")

# ✅ FIX: correct elbow detection
# Original code had +1 off-by-one: it picked the K AFTER the biggest drop
# instead of AT the biggest drop.
# The elbow is the K where the drop from K-1 to K is largest.
# deltas[i] = inertia[i] - inertia[i+1]  → biggest delta at index i
# → elbow is at k_range[i+1] (the K that caused the drop)
# Example: if biggest drop is between K=3 and K=4, elbow = K=4
k_list        = list(k_range)
deltas        = [inertias[i] - inertias[i+1] for i in range(len(inertias) - 1)]
drop_idx      = deltas.index(max(deltas))        # index where biggest drop starts
elbow_k       = k_list[drop_idx + 1]             # K at the bottom of that drop
elbow_inertia = inertias[drop_idx + 1]
# --- Plot elbow curve ---
plt.figure(figsize=(9, 5))
plt.plot(list(k_range), inertias,
         marker="o", color="#2c5f8a", linewidth=2,
         markersize=7, markerfacecolor="white", markeredgewidth=2)
plt.axvline(x=elbow_k, color="#c0392b", linestyle="--",
            linewidth=1.5, label=f"Suggested elbow: K={elbow_k}")
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
print(f"  Suggested K from elbow : {elbow_k}")


# ==========================================
# 3️⃣ SILHOUETTE SCORE — Confirm K
# ==========================================
# Silhouette score: how well each point fits
# its own cluster vs the nearest other cluster.
# Range: -1 (wrong cluster) → 0 → +1 (perfect fit)
# Higher = better separated, more distinct clusters.
# ==========================================

print("\nComputing Silhouette Scores (K = 2 to 7)...")

sil_scores = {}

for k in range(2, 8):
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_num)
    score  = silhouette_score(
        X_num, labels,
        sample_size  = min(1000, len(X_num)),  # ✅ safe for small datasets
        random_state = RANDOM_STATE
    )
    sil_scores[k] = score
    print(f"  K={k}  Silhouette={score:.4f}")

best_k_sil = max(sil_scores, key=sil_scores.get)
print(f"\n  Best K from silhouette : {best_k_sil}  (score={sil_scores[best_k_sil]:.4f})")
# After computing best_k_sil...

# ✅ FIX: if all silhouette scores are low (< 0.25), 
#         clusters are weak regardless — prefer elbow K for richer segmentation
MAX_SIL = max(sil_scores.values())
if MAX_SIL < 0.25 or best_k_sil == 2:
    # Silhouette is weak — elbow gives more business-meaningful segments
    FINAL_K = max(elbow_k, 3)   # at least 3 segments for business use
    print(f"  ⚠️  Silhouette scores weak (max={MAX_SIL:.3f}) — using K={FINAL_K} (elbow/floor)")
else:
    FINAL_K = best_k_sil

# --- Plot silhouette scores ---
plt.figure(figsize=(8, 4))
ks     = list(sil_scores.keys())
scores = list(sil_scores.values())
bars   = plt.bar(
    ks, scores,
    color=["#c0392b" if k == best_k_sil else "#2c5f8a" for k in ks],
    edgecolor="white"
)
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

# Select final K (silhouette is more reliable than elbow heuristic)
print(f"\n  Final K selected : {FINAL_K}")


# ==========================================
# 4️⃣ TRAIN FINAL K-MEANS
# ==========================================

print(f"\nTraining K-Means with K={FINAL_K}...")

kmeans = KMeans(
    n_clusters = FINAL_K,
    random_state = RANDOM_STATE,
    n_init     = 10,
    max_iter   = 300,
)
cluster_labels = kmeans.fit_predict(X_num)

print(f"  ✅ K-Means trained")
print(f"  Cluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, n in zip(unique, counts):
    pct = n / len(cluster_labels) * 100
    print(f"    Cluster {c} : {n} customers  ({pct:.1f}%)")

joblib.dump(kmeans, "models/kmeans_model.pkl")
# ==========================================
# 🔍 SILHOUETTE ANALYSIS (DETAILED)
# ==========================================

from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

sil_values = silhouette_samples(X_num, cluster_labels)
print("  ✅ models/kmeans_model.pkl saved")


# ==========================================
# 5️⃣ PCA 2D VISUALIZATION
# ==========================================

print("\nVisualizing clusters with PCA 2D...")

pca_2d   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X_num)

var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
var_pc2 = pca_2d.explained_variance_ratio_[1] * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left panel: colored by cluster ---
ax = axes[0]
for i in range(FINAL_K):
    mask = cluster_labels == i
    ax.scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c     = PALETTE[i % len(PALETTE)],
        label = f"Cluster {i}",
        alpha = 0.45, s=15, edgecolors="none"
    )
ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=10)
ax.set_title("K-Means Clusters (PCA 2D)", fontsize=12)
ax.legend(markerscale=2, fontsize=9)

# --- Right panel: colored by actual Churn ---
# ✅ FIX: y_all already has reset_index from concat(ignore_index=True)
#         so y_all.values aligns with X_pca_2d rows correctly
ax2 = axes[1]
churn_colors = {0: "#2c5f8a", 1: "#c0392b"}
churn_labels = {0: "Fidèle (0)", 1: "Churn (1)"}

y_vals = y_all.values   # numpy array, aligned with X_pca_2d

for label, color in churn_colors.items():
    mask = y_vals == label
    ax2.scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c     = color,
        label = churn_labels[label],
        alpha = 0.35, s=15, edgecolors="none"
    )
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
# 6️⃣ CLUSTER PROFILING (Inverse-scaled)
# ==========================================
# ✅ FIX: X_train.csv contains StandardScaler values
#    (mean=0, std=1) — not readable for business interpretation.
#    We inverse_transform the numeric columns to recover
#    original units (£ for monetary, days for tenure, etc.)
#    before computing cluster means.
# ==========================================

print("\nProfiling clusters (inverse-scaling for readability)...")

# Inverse-transform each clustering column individually.
# We cannot pass the full scaler.inverse_transform() because it expects
# all 54 columns it was fitted on. Instead we manually apply:
#   original_value = scaled_value * std + mean
# using the scaler's stored mean_ and scale_ arrays.
scaler_cols  = list(scaler.feature_names_in_)   # all 54 column names
X_inv        = X_num.copy()

for col in X_num.columns:
    if col in scaler_cols:
        idx            = scaler_cols.index(col)
        mean_val       = scaler.mean_[idx]
        std_val        = scaler.scale_[idx]
        X_inv[col]     = X_num[col] * std_val + mean_val
    # if col not in scaler_cols it was never scaled — leave as-is

# Build profile dataframe
X_profile            = X_inv.copy()
X_profile["Cluster"] = cluster_labels
X_profile["Churn"]   = y_all.values   # ✅ already reset-indexed

# ✅ FIX: correct column names matching preprocessing_v3.py output
# ✅ FIX: "Churn" removed from profile_features — added separately below
# ✅ FIX: "Recency" removed — dropped as leaky in preprocessing
CANDIDATE_PROFILE_FEATURES = [
    "Frequency",
    "MonetaryTotal",
    "CustomerTenureDays",
    "ReturnRatio",
    "CancelledTransactions",
    "UniqueProducts",
    "SupportTicketsCount",     # ✅ was "SupportTickets" → wrong name
    "EngagementScore",
    "DisengagementScore",
    "AvgBasketValue",
    "Country_TargetEnc",
]

# Only keep columns that actually exist in the data
profile_features = [f for f in CANDIDATE_PROFILE_FEATURES if f in X_profile.columns]
print(f"  Profiling on {len(profile_features)} features: {profile_features}")

# Add Churn separately (not part of profile_features to avoid duplication)
profile_agg = (
    X_profile
    .groupby("Cluster")[profile_features + ["Churn"]]
    .mean()
    .round(3)
)

print("\n  Cluster Profiles (original scale, mean values):")
print(profile_agg.to_string())


# ==========================================
# 7️⃣ CLUSTER HEATMAP
# ==========================================

profile_plot = profile_agg.drop(columns=["Churn"], errors="ignore")

# Normalize per column for heatmap colour scale
profile_norm = (
    (profile_plot - profile_plot.min())
    / (profile_plot.max() - profile_plot.min() + 1e-9)
)

plt.figure(figsize=(max(10, len(profile_features) + 2), max(4, FINAL_K + 2)))
sns.heatmap(
    profile_norm,
    annot        = profile_plot.round(2),
    fmt          = ".2f",
    cmap         = "YlOrRd",
    linewidths   = 0.5,
    cbar_kws     = {"label": "Normalized value (0=min, 1=max)"},
    yticklabels  = [f"Cluster {i}" for i in range(FINAL_K)],
)
plt.title("Cluster Profiles — Mean Feature Values (original scale)", fontsize=13)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("reports/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/cluster_profiles.png saved")


# ==========================================
# 8️⃣ CHURN RATE PER CLUSTER
# ==========================================

churn_per_cluster = profile_agg["Churn"] * 100   # convert to %

plt.figure(figsize=(8, 4))
bars = plt.bar(
    [f"Cluster {i}" for i in churn_per_cluster.index],
    churn_per_cluster.values,
    color       = PALETTE[:FINAL_K],
    edgecolor   = "white",
)
for bar, val in zip(bars, churn_per_cluster.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold"
    )

overall_churn = y_all.mean() * 100
plt.axhline(
    y       = overall_churn,
    color   = "#888",
    linestyle = "--",
    linewidth = 1.2,
    label   = f"Overall churn rate ({overall_churn:.1f}%)"
)
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate per Cluster", fontsize=13)
plt.legend()
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("reports/cluster_churn_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/cluster_churn_rate.png saved")


# ==========================================
# 9️⃣ CLUSTER NAMING (Business Interpretation)
# ==========================================
# ✅ FIX: auto-naming now uses Frequency and MonetaryTotal
#    instead of Recency (which was dropped).
#    Logic: churn rate + frequency + monetary → business label.
# ==========================================

print("\nInterpreting clusters...")

cluster_names = {}
used_labels   = set()   # prevent duplicate names across clusters

has_freq = "Frequency"       in profile_agg.columns
has_mon  = "MonetaryTotal"   in profile_agg.columns
has_eng  = "EngagementScore" in profile_agg.columns

freq_median = profile_agg["Frequency"].median()     if has_freq else 0
mon_median  = profile_agg["MonetaryTotal"].median() if has_mon  else 0

# Sort clusters by churn rate descending so highest-risk gets named first
# and the labels are assigned in a meaningful priority order.
sorted_clusters = profile_agg["Churn"].sort_values(ascending=False).index.tolist()

LABEL_PRIORITY = [
    # (condition_fn,  label,        description)
    (lambda r, f, m: r > 60,                       "Lost / Churned", "Very high churn, low engagement"),
    (lambda r, f, m: r > 35,                       "At Risk",        "Above-average churn, moderate activity"),
    (lambda r, f, m: f > freq_median and m > mon_median, "Champions", "High frequency + high spend, low churn"),
    (lambda r, f, m: f > freq_median,              "Loyal",          "Frequent buyers, moderate spend"),
    (lambda r, f, m: True,                         "Occasional",     "Low frequency, variable spend"),
]

for i in sorted_clusters:
    churn_rate = churn_per_cluster[i]
    frequency  = profile_agg.loc[i, "Frequency"]      if has_freq else 0
    monetary   = profile_agg.loc[i, "MonetaryTotal"]   if has_mon  else 0

    for condition, label, desc in LABEL_PRIORITY:
        if condition(churn_rate, frequency, monetary) and label not in used_labels:
            cluster_names[i] = f"Cluster {i} — {label}"
            used_labels.add(label)
            break
    else:
        # Fallback if all preferred labels are taken
        cluster_names[i] = f"Cluster {i} — Segment {i}"
        desc = "No distinct label available"
    size = counts[i]
    name = cluster_names[i]
    print(f"  {name}")
    print(f"    → Churn: {churn_per_cluster[i]:.1f}%  |  Size: {counts[i]} customers")
    print()


# ==========================================
# 🔟 SAVE RESULTS
# ==========================================

print("Saving cluster assignments...")

X_all_with_clusters              = X_all.copy()
X_all_with_clusters["Cluster"]   = cluster_labels
X_all_with_clusters["ClusterName"] = [
    cluster_names[c].split("—")[-1].strip() for c in cluster_labels
]
X_all_with_clusters["Churn"]     = y_all.values

X_all_with_clusters.to_csv(
    "data/processed/customers_with_clusters.csv",
    index=False
)
print("  ✅ data/processed/customers_with_clusters.csv saved")

profile_agg.to_csv("reports/cluster_profiles_summary.csv")
print("  ✅ reports/cluster_profiles_summary.csv saved")


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*55)
print("  CLUSTERING COMPLETE")
print("="*55)
print(f"  Algorithm   : K-Means")
k_method = "silhouette" if FINAL_K == best_k_sil else f"elbow override (sil. weak, max={MAX_SIL:.3f})"
print(f"  K selected  : {FINAL_K} clusters  ({k_method})")
print(f"  Dataset     : {len(X_all)} customers")
print(f"  Elbow K     : {elbow_k}  |  Silhouette K: {best_k_sil}")
print(f"\n  Cluster summary:")
for i in range(FINAL_K):
    print(f"    {cluster_names[i]:<45}  {churn_per_cluster[i]:.1f}% churn  ({counts[i]} customers)")
print(f"\n  Reports generated:")
print("    reports/elbow_curve.png")
print("    reports/silhouette_scores.png")
print("    reports/clusters_pca2d.png")
print("    reports/cluster_profiles.png")
print("    reports/cluster_churn_rate.png")
print("    reports/cluster_profiles_summary.csv")
print("    data/processed/customers_with_clusters.csv")
print("="*55)
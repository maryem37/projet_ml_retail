# ==========================================
# RETAIL ML PROJECT - CLUSTERING SCRIPT
# ==========================================
# Unsupervised learning — K-Means Clustering
# Goal: Discover natural customer segments
# without using the Churn label.
#
# FIXES vs previous version:
#   ✅ Removed "Recency" from profile features — dropped as leaky
#   ✅ Fixed column names: SatisfactionScore, SupportTicketsCount
#   ✅ Fixed elbow heuristic off-by-one error
#   ✅ Profiling uses inverse-scaled values (readable business numbers)
#   ✅ Removed "Churn" duplicate from profile_features list
#   ✅ Cluster auto-naming no longer references Recency
#   ✅ y_all index alignment fixed with reset_index
#   ✅ Profile features guarded with existence check
#
# FIXES in this version (v2):
#   ✅ sil_values now used for per-cluster silhouette bar plot
#      (was computed but silently unused before)
#   ✅ counts[i] replaced by cluster_sizes dict — robust to non-contiguous
#      cluster label ordering (np.unique order != sorted_clusters order)
#   ✅ size variable now printed consistently in cluster naming block
#   ✅ KMeans reuse: if FINAL_K == best_k_sil, the already-fitted KMeans
#      from the silhouette loop is reused — avoids a redundant refit
#   ✅ silhouette_samples import moved to top-level (was mid-script)
#
# Steps:
#   1. Load preprocessed data
#   2. Elbow method → find optimal K
#   3. Silhouette score → confirm K
#   4. Train K-Means
#   5. Visualize clusters (PCA 2D)
#   6. Per-cluster silhouette plot
#   7. Profile each cluster (inverse-scaled values)
#   8. Save results
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.cluster  import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics  import silhouette_score, silhouette_samples


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

# Use a curated subset of interpretable features for clustering.
# All features are already StandardScaler-transformed (mean=0, std=1).
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

print(f"  Clustering on {len(clustering_cols)} interpretable features: {clustering_cols}")
print(f"  Numeric features used          : {X_num.shape[1]}")
print(f"  Note: features are StandardScaler-transformed (mean=0, std=1)")

# Remove outliers before clustering.
# Customers with any feature beyond 4 standard deviations are data anomalies
# that K-Means will isolate into tiny meaningless clusters.
z_scores     = np.abs(X_num)   # already z-scores (StandardScaler output)
outlier_mask = (z_scores > 4).any(axis=1)
n_outliers   = outlier_mask.sum()
print(f"  Outliers removed (|z| > 4)     : {n_outliers} customers")

X_num = X_num[~outlier_mask].reset_index(drop=True)
y_all = y_all[~outlier_mask].reset_index(drop=True)
X_all = X_all[~outlier_mask].reset_index(drop=True)
print(f"  Customers after outlier removal: {len(X_num)}")


# ==========================================
# 2️⃣ ELBOW METHOD — Find optimal K
# ==========================================

print("\nRunning Elbow Method (K = 2 to 10)...")

k_range  = range(2, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_num)
    inertias.append(km.inertia_)
    print(f"  K={k}  Inertia={km.inertia_:,.0f}")

# Elbow = K at the bottom of the biggest inertia drop.
# deltas[i] = inertia[i] - inertia[i+1]
# drop_idx  = index where the biggest drop starts (between K[drop_idx] and K[drop_idx+1])
# elbow_k   = K at the bottom of that drop = k_list[drop_idx + 1]
k_list        = list(k_range)
deltas        = [inertias[i] - inertias[i+1] for i in range(len(inertias) - 1)]
drop_idx      = deltas.index(max(deltas))
elbow_k       = k_list[drop_idx + 1]
elbow_inertia = inertias[drop_idx + 1]

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

print("\nComputing Silhouette Scores (K = 2 to 7)...")

sil_scores   = {}
# FIX v2: store fitted KMeans objects so we can reuse the best one in step 4
# instead of refitting from scratch (avoids a redundant KMeans.fit call).
sil_km_cache = {}

for k in range(2, 8):
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_num)
    score  = silhouette_score(
        X_num, labels,
        sample_size  = min(1000, len(X_num)),
        random_state = RANDOM_STATE
    )
    sil_scores[k]   = score
    sil_km_cache[k] = (km, labels)   # ✅ cache — reused in step 4 if k == FINAL_K
    print(f"  K={k}  Silhouette={score:.4f}")

best_k_sil = max(sil_scores, key=sil_scores.get)
print(f"\n  Best K from silhouette : {best_k_sil}  (score={sil_scores[best_k_sil]:.4f})")

MAX_SIL = max(sil_scores.values())
if MAX_SIL < 0.25 or best_k_sil == 2:
    FINAL_K = max(elbow_k, 3)
    print(f"  ⚠️  Silhouette scores weak (max={MAX_SIL:.3f}) — using K={FINAL_K} (elbow/floor)")
else:
    FINAL_K = best_k_sil

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
print(f"\n  Final K selected : {FINAL_K}")


# ==========================================
# 4️⃣ TRAIN FINAL K-MEANS
# ==========================================

print(f"\nTraining K-Means with K={FINAL_K}...")

# FIX v2: reuse cached KMeans if FINAL_K was already fitted in the silhouette loop.
if FINAL_K in sil_km_cache:
    kmeans, cluster_labels = sil_km_cache[FINAL_K]
    print(f"  ✅ Reusing cached KMeans(K={FINAL_K}) from silhouette loop")
else:
    kmeans = KMeans(
        n_clusters   = FINAL_K,
        random_state = RANDOM_STATE,
        n_init       = 10,
        max_iter     = 300,
    )
    cluster_labels = kmeans.fit_predict(X_num)
    print(f"  ✅ K-Means fitted (K={FINAL_K} outside silhouette range)")

print(f"  Cluster distribution:")

# FIX v2: build cluster_sizes dict — robust to any label ordering.
# np.unique returns (unique_values, counts) sorted by value.
# counts[i] corresponds to unique[i], NOT necessarily to cluster i directly.
# Using a dict avoids silent index mismatches when iterating in a different order.
unique, raw_counts = np.unique(cluster_labels, return_counts=True)
cluster_sizes      = dict(zip(unique, raw_counts))   # {cluster_id: count}

for c, n in cluster_sizes.items():
    pct = n / len(cluster_labels) * 100
    print(f"    Cluster {c} : {n} customers  ({pct:.1f}%)")

joblib.dump(kmeans, "models/kmeans_model.pkl")
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

ax2 = axes[1]
churn_colors = {0: "#2c5f8a", 1: "#c0392b"}
churn_labels = {0: "Fidèle (0)", 1: "Churn (1)"}
y_vals       = y_all.values

for label, color in churn_colors.items():
    mask = y_vals == label
    ax2.scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c=color, label=churn_labels[label],
        alpha=0.35, s=15, edgecolors="none"
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
# 6️⃣ PER-CLUSTER SILHOUETTE PLOT
# ==========================================
# FIX v2: sil_values was computed but never used before.
# This plot shows the silhouette coefficient distribution per cluster,
# which reveals whether individual clusters are well-separated or blended.
# A cluster with many negative silhouette values is poorly defined.
# ==========================================

print("\nGenerating per-cluster silhouette plot...")

sil_values = silhouette_samples(X_num, cluster_labels)

fig, ax = plt.subplots(figsize=(9, max(4, FINAL_K * 1.5)))
y_lower = 10

for i in range(FINAL_K):
    cluster_sil = np.sort(sil_values[cluster_labels == i])
    size_i      = cluster_sizes[i]
    y_upper     = y_lower + size_i

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0, cluster_sil,
        facecolor=PALETTE[i % len(PALETTE)],
        edgecolor="none",
        alpha=0.8,
        label=f"Cluster {i}  (n={size_i})",
    )
    ax.text(
        -0.05, y_lower + size_i / 2,
        f"C{i}", ha="right", va="center", fontsize=9, fontweight="bold"
    )
    y_lower = y_upper + 10   # gap between clusters

mean_sil = sil_values.mean()
ax.axvline(mean_sil, color="#c0392b", linestyle="--", linewidth=1.5,
           label=f"Mean silhouette = {mean_sil:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Per-cluster Silhouette Plot — K={FINAL_K}", fontsize=13)
ax.set_xlim(-0.3, 1.0)
ax.set_yticks([])
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("reports/silhouette_per_cluster.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/silhouette_per_cluster.png saved")


# ==========================================
# 7️⃣ CLUSTER PROFILING (Inverse-scaled)
# ==========================================
# X_train.csv contains StandardScaler values (mean=0, std=1).
# We manually inverse-transform using scaler.mean_ and scaler.scale_
# to recover original units before computing cluster means.
# We cannot use scaler.inverse_transform() directly because it expects
# all 54 columns it was fitted on.
# ==========================================

print("\nProfiling clusters (inverse-scaling for readability)...")

scaler_cols = list(scaler.feature_names_in_)
X_inv       = X_num.copy()

for col in X_num.columns:
    if col in scaler_cols:
        idx          = scaler_cols.index(col)
        X_inv[col]   = X_num[col] * scaler.scale_[idx] + scaler.mean_[idx]
    # columns not in scaler_cols were never scaled — leave as-is

X_profile              = X_inv.copy()
X_profile["Cluster"]   = cluster_labels
X_profile["Churn"]     = y_all.values

CANDIDATE_PROFILE_FEATURES = [
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

profile_features = [f for f in CANDIDATE_PROFILE_FEATURES if f in X_profile.columns]
print(f"  Profiling on {len(profile_features)} features: {profile_features}")

profile_agg = (
    X_profile
    .groupby("Cluster")[profile_features + ["Churn"]]
    .mean()
    .round(3)
)

print("\n  Cluster Profiles (original scale, mean values):")
print(profile_agg.to_string())


# ==========================================
# 8️⃣ CLUSTER HEATMAP
# ==========================================

profile_plot = profile_agg.drop(columns=["Churn"], errors="ignore")

profile_norm = (
    (profile_plot - profile_plot.min())
    / (profile_plot.max() - profile_plot.min() + 1e-9)
)

plt.figure(figsize=(max(10, len(profile_features) + 2), max(4, FINAL_K + 2)))
sns.heatmap(
    profile_norm,
    annot       = profile_plot.round(2),
    fmt         = ".2f",
    cmap        = "YlOrRd",
    linewidths  = 0.5,
    cbar_kws    = {"label": "Normalized value (0=min, 1=max)"},
    yticklabels = [f"Cluster {i}" for i in range(FINAL_K)],
)
plt.title("Cluster Profiles — Mean Feature Values (original scale)", fontsize=13)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("reports/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ reports/cluster_profiles.png saved")


# ==========================================
# 9️⃣ CHURN RATE PER CLUSTER
# ==========================================

churn_per_cluster = profile_agg["Churn"] * 100   # convert to %

plt.figure(figsize=(8, 4))
bars = plt.bar(
    [f"Cluster {i}" for i in churn_per_cluster.index],
    churn_per_cluster.values,
    color     = PALETTE[:FINAL_K],
    edgecolor = "white",
)
for bar, val in zip(bars, churn_per_cluster.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold"
    )

overall_churn = y_all.mean() * 100
plt.axhline(
    y         = overall_churn,
    color     = "#888",
    linestyle = "--",
    linewidth = 1.2,
    label     = f"Overall churn rate ({overall_churn:.1f}%)"
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
# 🔟 CLUSTER NAMING (Business Interpretation)
# ==========================================
# Auto-naming uses Frequency and MonetaryTotal (Recency was dropped as leaky).
# Clusters are sorted by churn rate descending so highest-risk gets priority.
# FIX v2: cluster_sizes[i] used instead of counts[i] — robust dict lookup.
# FIX v2: size variable is now printed consistently in the output block.
# ==========================================

print("\nInterpreting clusters...")

cluster_names = {}
used_labels   = set()

has_freq = "Frequency"     in profile_agg.columns
has_mon  = "MonetaryTotal" in profile_agg.columns

freq_median = profile_agg["Frequency"].median()     if has_freq else 0
mon_median  = profile_agg["MonetaryTotal"].median() if has_mon  else 0

sorted_clusters = profile_agg["Churn"].sort_values(ascending=False).index.tolist()

LABEL_PRIORITY = [
    (lambda r, f, m: r > 60,                              "Lost / Churned", "Very high churn, low engagement"),
    (lambda r, f, m: r > 35,                              "At Risk",        "Above-average churn, moderate activity"),
    (lambda r, f, m: f > freq_median and m > mon_median,  "Champions",      "High frequency + high spend, low churn"),
    (lambda r, f, m: f > freq_median,                     "Loyal",          "Frequent buyers, moderate spend"),
    (lambda r, f, m: True,                                "Occasional",     "Low frequency, variable spend"),
]

for i in sorted_clusters:
    churn_rate = churn_per_cluster[i]
    frequency  = profile_agg.loc[i, "Frequency"]    if has_freq else 0
    monetary   = profile_agg.loc[i, "MonetaryTotal"] if has_mon  else 0

    for condition, label, desc in LABEL_PRIORITY:
        if condition(churn_rate, frequency, monetary) and label not in used_labels:
            cluster_names[i] = f"Cluster {i} — {label}"
            used_labels.add(label)
            break
    else:
        cluster_names[i] = f"Cluster {i} — Segment {i}"
        desc = "No distinct label available"

    # FIX v2: use cluster_sizes dict (robust) and print size explicitly
    size = cluster_sizes[i]
    name = cluster_names[i]
    print(f"  {name}")
    print(f"    → Churn: {churn_per_cluster[i]:.1f}%  |  Size: {size} customers  |  {desc}")
    print()


# ==========================================
# 1️⃣1️⃣ SAVE RESULTS
# ==========================================

print("Saving cluster assignments...")

X_all_with_clusters                = X_all.copy()
X_all_with_clusters["Cluster"]     = cluster_labels
X_all_with_clusters["ClusterName"] = [
    cluster_names[c].split("—")[-1].strip() for c in cluster_labels
]
X_all_with_clusters["Churn"]       = y_all.values

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
    size = cluster_sizes[i]
    print(f"    {cluster_names[i]:<45}  {churn_per_cluster[i]:.1f}% churn  ({size} customers)")
print(f"\n  Reports generated:")
print("    reports/elbow_curve.png")
print("    reports/silhouette_scores.png")
print("    reports/silhouette_per_cluster.png")
print("    reports/clusters_pca2d.png")
print("    reports/cluster_profiles.png")
print("    reports/cluster_churn_rate.png")
print("    reports/cluster_profiles_summary.csv")
print("    data/processed/customers_with_clusters.csv")
print("="*55)
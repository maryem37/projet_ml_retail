# ==========================================
# RETAIL ML PROJECT - UTILS SCRIPT
# ==========================================
# Contains reusable analysis functions:
#   - Correlation heatmap
#   - Multicollinearity removal (VIF)
#   - PCA (ACP) analysis & visualization
#   - Feature importance plot
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1️⃣ CORRELATION HEATMAP
# ==========================================

def plot_correlation_heatmap(df, title="Correlation Matrix", save_path="reports/correlation_heatmap.png"):
    """
    Plots and saves a heatmap of feature correlations.
    Useful for spotting multicollinearity visually.

    Parameters:
        df        : DataFrame (numeric columns only)
        title     : Plot title
        save_path : Where to save the image
    """
    print("Plotting correlation heatmap...")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(18, 14))
    corr_matrix = numeric_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title, fontsize=14, pad=16)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {save_path}")

    return corr_matrix


# ==========================================
# 2️⃣ MULTICOLLINEARITY REMOVAL (Threshold)
# ==========================================

def remove_multicollinear_features(df, threshold=0.8, exclude_cols=None):
    """
    Removes one of each pair of features with |correlation| > threshold.
    Keeps the first feature encountered (you can adjust based on business logic).

    Parameters:
        df           : DataFrame
        threshold    : Correlation threshold (default 0.8)
        exclude_cols : List of columns to never drop (e.g. target)

    Returns:
        df_clean     : DataFrame with correlated features removed
        dropped      : List of dropped column names
    """
    print(f"\nChecking multicollinearity (threshold = {threshold})...")

    if exclude_cols is None:
        exclude_cols = []

    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

    # Remove excluded cols from analysis
    analysis_cols = [c for c in numeric_df.columns if c not in exclude_cols]
    corr_matrix   = numeric_df[analysis_cols].corr().abs()

    # Upper triangle only
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation above threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"  ⚠️  Dropping {len(to_drop)} multicollinear features:")
        for col in to_drop:
            # Show which feature it's correlated with
            correlated_with = upper.index[upper[col] > threshold].tolist()
            print(f"      - {col}  ←→  {correlated_with}")
        df = df.drop(columns=to_drop, errors="ignore")
    else:
        print("  ✅ No multicollinear features found above threshold.")

    return df, to_drop


# ==========================================
# 3️⃣ VIF — Variance Inflation Factor
# ==========================================

def compute_vif(df, max_features=30):
    """
    Computes VIF for numeric features.
    VIF > 10 → severe multicollinearity.
    VIF > 5  → moderate multicollinearity.

    Parameters:
        df           : DataFrame (numeric only)
        max_features : Limit columns to avoid memory issues

    Returns:
        vif_df : DataFrame with VIF scores sorted descending
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  ⚠️  statsmodels not installed. Run: pip install statsmodels")
        return None

    print("\nComputing VIF scores...")

    numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

    # Limit features for performance
    cols = numeric_df.columns[:max_features]
    X    = numeric_df[cols].values

    vif_data = pd.DataFrame()
    vif_data["Feature"] = cols
    vif_data["VIF"]     = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_data            = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    print(vif_data.to_string(index=False))

    high_vif = vif_data[vif_data["VIF"] > 10]
    if len(high_vif) > 0:
        print(f"\n  ⚠️  {len(high_vif)} features with VIF > 10 (severe multicollinearity)")
    else:
        print("\n  ✅ No severe multicollinearity detected (all VIF ≤ 10)")

    return vif_data


# ==========================================
# 4️⃣ PCA — ANALYSE EN COMPOSANTES PRINCIPALES
# ==========================================

def run_pca(X_train, X_test, n_components=None, variance_threshold=0.95,
            save_path="reports/pca_variance.png"):
    """
    Applies PCA to reduce dimensionality.
    Automatically selects number of components to retain
    `variance_threshold` (default 95%) of variance.

    Parameters:
        X_train            : Training features (already scaled)
        X_test             : Test features (already scaled)
        n_components       : Fixed number of components (overrides variance_threshold)
        variance_threshold : % of variance to retain (0.0 - 1.0)
        save_path          : Path to save the explained variance plot

    Returns:
        X_train_pca : Transformed training data
        X_test_pca  : Transformed test data
        pca         : Fitted PCA object (for inverse_transform later)
        n_comp      : Number of components selected
    """
    print(f"\nRunning PCA (variance threshold = {variance_threshold*100:.0f}%)...")

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_tr_num     = X_train[numeric_cols].values
    X_te_num     = X_test[numeric_cols].values

    # Step 1: Fit PCA with all components to find optimal n
    pca_full = PCA(random_state=42)
    pca_full.fit(X_tr_num)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        n_comp = np.argmax(cumulative_variance >= variance_threshold) + 1
    else:
        n_comp = n_components

    print(f"  Components selected : {n_comp} (from {X_tr_num.shape[1]} original features)")
    print(f"  Variance retained   : {cumulative_variance[n_comp-1]*100:.2f}%")

    # Step 2: Fit final PCA with selected n_components
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_tr_num)
    X_test_pca  = pca.transform(X_te_num)

    # Wrap as DataFrame
    pc_cols     = [f"PC{i+1}" for i in range(n_comp)]
    X_train_pca = pd.DataFrame(X_train_pca, columns=pc_cols, index=X_train.index)
    X_test_pca  = pd.DataFrame(X_test_pca,  columns=pc_cols, index=X_test.index)

    # Step 3: Plot explained variance
    _plot_pca_variance(pca_full, n_comp, save_path)

    return X_train_pca, X_test_pca, pca, n_comp


def _plot_pca_variance(pca_full, n_comp_selected, save_path):
    """Plots individual and cumulative explained variance."""

    explained   = pca_full.explained_variance_ratio_[:40] * 100
    cumulative  = np.cumsum(pca_full.explained_variance_ratio_[:40]) * 100
    components  = range(1, len(explained) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart — individual variance
    colors = ["#2c5f8a" if i < n_comp_selected else "#d0cfc9" for i in range(len(explained))]
    ax1.bar(components, explained, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Individual Explained Variance per Component")
    ax1.axvline(x=n_comp_selected + 0.5, color="#c0392b", linestyle="--", linewidth=1.5,
                label=f"Selected: {n_comp_selected} components")
    ax1.legend(fontsize=9)

    # Line chart — cumulative variance
    ax2.plot(components, cumulative, color="#2c5f8a", linewidth=2, marker="o", markersize=3)
    ax2.axhline(y=95, color="#e67e22", linestyle="--", linewidth=1.2, label="95% threshold")
    ax2.axvline(x=n_comp_selected, color="#c0392b", linestyle="--", linewidth=1.5,
                label=f"Selected: {n_comp_selected} components")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.fill_between(components, cumulative, alpha=0.08, color="#2c5f8a")

    plt.suptitle("PCA — Explained Variance Analysis", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ PCA variance plot saved → {save_path}")


def plot_pca_2d(X_pca, y, save_path="reports/pca_2d.png"):
    """
    Visualizes the first 2 principal components colored by Churn label.

    Parameters:
        X_pca     : DataFrame with PC1, PC2 columns (at minimum)
        y         : Target series (Churn 0/1)
        save_path : Path to save the plot
    """
    print("  Plotting PCA 2D scatter...")

    plt.figure(figsize=(9, 6))

    colors  = {0: "#2c5f8a", 1: "#c0392b"}
    labels  = {0: "Fidèle (0)", 1: "Churn (1)"}

    for label, color in colors.items():
        mask = y.values == label
        plt.scatter(
            X_pca["PC1"].values[mask],
            X_pca["PC2"].values[mask],
            c=color, label=labels[label],
            alpha=0.45, s=18, edgecolors="none"
        )

    plt.xlabel("PC1", fontsize=11)
    plt.ylabel("PC2", fontsize=11)
    plt.title("PCA — 2D Projection (PC1 vs PC2)", fontsize=13)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ PCA 2D scatter saved → {save_path}")


# ==========================================
# 5️⃣ FEATURE IMPORTANCE PLOT (standalone)
# ==========================================

def plot_feature_importance(model, feature_names, top_n=20,
                             save_path="reports/feature_importance.png"):
    """
    Plots top N feature importances from a tree-based model.

    Parameters:
        model         : Trained tree-based model (RandomForest, GradientBoosting)
        feature_names : List of feature names
        top_n         : Number of top features to display
        save_path     : Path to save the plot
    """
    if not hasattr(model, "feature_importances_"):
        print("  ⚠️  Model does not have feature_importances_ attribute.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="#2c5f8a", edgecolor="white")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Feature importance saved → {save_path}")


# ==========================================
# 6️⃣ QUICK DATA QUALITY REPORT
# ==========================================

def data_quality_report(df):
    """
    Prints a quick summary of data quality:
    missing values, unique counts, dtypes.
    """
    print("\n" + "="*55)
    print("  DATA QUALITY REPORT")
    print("="*55)
    print(f"  Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        print(f"\n  Missing values ({len(missing)} columns):")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    {col:<35} {count:>5} ({pct:.1f}%)")
    else:
        print("\n  ✅ No missing values.")

    print(f"\n  Numeric columns  : {len(df.select_dtypes(include='number').columns)}")
    print(f"  Categoric columns: {len(df.select_dtypes(include='object').columns)}")
    print("="*55 + "\n")


# ==========================================
# MAIN — Run all analysis on raw data
# ==========================================

if __name__ == "__main__":

    print("Loading data for analysis...")
    df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

    # 1. Data quality report
    data_quality_report(df)

    # 2. Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plot_correlation_heatmap(numeric_df, title="Raw Data — Correlation Matrix")

    # 3. Multicollinearity check
    df_clean, dropped = remove_multicollinear_features(
        numeric_df,
        threshold=0.8,
        exclude_cols=["Churn"]
    )
    print(f"\n  Features remaining after multicollinearity removal: {df_clean.shape[1]}")

    # 4. VIF
    compute_vif(df_clean)

    # 5. PCA on scaled numeric data
    print("\nPreparing data for PCA...")
    from sklearn.model_selection import train_test_split

    X = df_clean.drop(columns=["Churn"], errors="ignore")
    y = df["Churn"] if "Churn" in df.columns else None

    X_filled = X.fillna(X.median(numeric_only=True))
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filled), columns=X_filled.columns)

    if y is not None:
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        X_train_pca, X_test_pca, pca, n_comp = run_pca(X_tr, X_te)
        plot_pca_2d(X_train_pca, y_tr)

    print("\n✅ Utils analysis complete. Check reports/ folder.")
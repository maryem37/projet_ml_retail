# ==========================================
# RETAIL ML PROJECT - CLUSTERING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# ==========================================
# 1️⃣ LOAD PROCESSED DATA
# ==========================================

print("Loading preprocessed data...")
X_train = pd.read_csv("data/train_test/X_train.csv")
X_test = pd.read_csv("data/train_test/X_test.csv")

# ==========================================
# 2️⃣ IMPUTATION DES VALEURS MANQUANTES
# ==========================================

print("Imputing missing values...")
imputer = SimpleImputer(strategy='median')  # médiane pour valeurs numériques/ordinales

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Vérification rapide
print("NaN par colonne (train):")
print(X_train_imputed.isna().sum())
print("NaN par colonne (test):")
print(X_test_imputed.isna().sum())

# ==========================================
# 3️⃣ PCA (DIMENSIONALITY REDUCTION)
# ==========================================

print("Applying PCA for dimensionality reduction...")
pca_components = 10  # changer selon vos besoins
pca = PCA(n_components=pca_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_imputed)
X_test_pca = pca.transform(X_test_imputed)

print(f"Explained variance by {pca_components} components: {np.sum(pca.explained_variance_ratio_):.2f}")

# ==========================================
# 4️⃣ KMEANS CLUSTERING
# ==========================================

print("Fitting KMeans clustering...")
k = 4  # nombre de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
train_clusters = kmeans.fit_predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)

# Ajouter les clusters comme nouvelle feature
X_train_imputed['Cluster'] = train_clusters
X_test_imputed['Cluster'] = test_clusters

# ==========================================
# 5️⃣ EVALUATION (SILHOUETTE SCORE)
# ==========================================

score = silhouette_score(X_train_pca, train_clusters)
print(f"Silhouette Score: {score:.3f}")

# ==========================================
# 6️⃣ VISUALISATION (OPTIONNEL)
# ==========================================

print("Visualizing first 2 PCA components with clusters...")
plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=train_clusters, cmap='viridis', s=50)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('KMeans Clusters on PCA-reduced data')
plt.show()

# ==========================================
# 7️⃣ SAVE CLUSTERED DATA
# ==========================================

print("Saving clustered datasets...")
X_train_imputed.to_csv("data/train_test/X_train_with_clusters.csv", index=False)
X_test_imputed.to_csv("data/train_test/X_test_with_clusters.csv", index=False)
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(pca, "models/pca_model.pkl")

print("✅ Clustering complete. Files saved in data/train_test/ and models/")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Charger le dataset
df = pd.read_csv('Credit_card_dataset.csv')

# Choisir les variables pour le clustering
X = df[['PURCHASES','CREDIT_LIMIT']].copy()

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering hiérarchique
Z = linkage(X_scaled, method='complete')
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=10.)
plt.title('Dendrogramme Hiérarchique')
plt.savefig('hierarchical_clusters.png')

# Définir 3 clusters hiérarchiques et ajouter au DataFrame
df['Hierarchical_Cluster'] = fcluster(Z, 3, criterion='maxclust')

# Clustering KMeans avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisation KMeans
plt.figure(figsize=(8,5))
plt.scatter(X['PURCHASES'], X['CREDIT_LIMIT'], c=df['KMeans_Cluster'], cmap='viridis')
plt.xlabel('PURCHASES')
plt.ylabel('CREDIT_LIMIT')
plt.title('Clusters KMeans')
plt.savefig('kmeans_clusters.png')

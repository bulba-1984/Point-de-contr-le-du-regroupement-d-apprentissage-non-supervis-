
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from ydata_profiling import ProfileReport
df = pd.read_csv('Credit_card_dataset.csv')
print(df.head())
print(df.info())
profile = ProfileReport(df, title='Credit Card Dataset Profiling', explorative=True)
profile.to_file('Credit_card_profiling.html')
X = df[['PURCHASES', 'CREDIT_LIMIT']]
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
agglo = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels_agglo = agglo.fit_predict(X_scaled)
df['cluster_agglo'] = labels_agglo
plt.figure(figsize=(8,6))
sns.scatterplot(x='PURCHASES', y='CREDIT_LIMIT', hue='cluster_agglo', palette='Set1', data=df)
plt.title('Clustering hiérarchique (Agglomératif)')
plt.savefig('agglo_clusters.png')
plt.clf()
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
df['cluster_kmeans'] = labels_kmeans
plt.figure(figsize=(8,6))
sns.scatterplot(x='PURCHASES', y='CREDIT_LIMIT', hue='cluster_kmeans', palette='Set2', data=df)
plt.title('Clustering partitionnel (KMeans)')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', marker='X', label='Centres')
plt.legend()
plt.savefig('kmeans_clusters.png')

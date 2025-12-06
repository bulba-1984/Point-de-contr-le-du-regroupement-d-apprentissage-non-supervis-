
# Credit Card Dataset - Clustering

## Description du projet
Segmentation client à partir de l'ensemble de données Kaggle pour identifier les groupes de clients à l'aide de clustering hiérarchique et KMeans.

## Contenu du dataset
- 8950 clients
- Colonnes : CUST_ID, BALANCE_FREQUENCY, PURCHASES, PAYMENTS, CREDIT_LIMIT, CASH_ADVANCE

## Étapes réalisées
1. Import et exploration des données
2. Profiling avec pandas_profiling
3. Gestion des valeurs manquantes et doublons
4. Encodage des variables catégorielles
5. Sélection des features pour clustering
6. Clustering hiérarchique et représentation graphique
7. Clustering KMeans et représentation graphique
8. Recherche de la meilleure valeur de k avec la méthode du coude

## Résultats
- Clusters identifiés avec les variables PURCHASES et CREDIT_LIMIT
- Rapports visuels sauvegardés sous 'hierarchical_clusters.png' et 'kmeans_clusters.png'

## Auteur
⚓ YFZ 2025 Bulba

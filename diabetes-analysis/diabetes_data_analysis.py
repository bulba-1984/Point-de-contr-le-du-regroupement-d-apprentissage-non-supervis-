#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploration et profilage des données de diabète
Usage: python3 diabetes_data_analysis.py <nom_du_fichier_csv>
"""

import sys
import pandas as pd

# Vérification des arguments
if len(sys.argv) < 2:
    print("❌ Aucun fichier CSV fourni. Usage: python3 diabetes_data_analysis.py <nom_du_fichier_csv>")
    sys.exit(1)

csv_file = sys.argv[1]

# Étape 1 : Chargement des données avec Pandas
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"❌ Fichier introuvable : {csv_file}")
    sys.exit(1)

print("=== 10 premières lignes du DataFrame ===")
print(df.head(), "\n")

print("=== Informations générales ===")
df.info()
print("\n=== Dimensions ===", df.shape)

# Identifier les colonnes avec des zéros ou valeurs manquantes
print("\n=== Valeurs manquantes par colonne ===")
print(df.isnull().sum())

print("\n=== Zéros par colonne (si pertinents) ===")
print((df == 0).sum())

# Analyse descriptive
print("\n=== Analyse descriptive ===")
print(df.describe())

# Étape 2 : Profilage avec ydata-profiling
try:
    from ydata_profiling import ProfileReport
except ImportError:
    print("❌ ydata-profiling non installé. Installez-le avec : pip install ydata-profiling")
    sys.exit(1)

print("\n=== Génération du rapport de profilage ydata-profiling ===")
profile = ProfileReport(df, title="Rapport de Profilage des Données Diabète", explorative=True)
report_file = "diabetes_profile_report.html"
profile.to_file(report_file)
print(f"✅ Rapport généré : {report_file}")

# Étape 3 : Mini-rapport résumé
print("\n=== Mini-rapport résumé ===")
critical_cols = ["Glucose", "Insulin", "BMI", "Age", "Outcome"]
for col in critical_cols:
    if col in df.columns:
        missing = df[col].isnull().sum()
        zeros = (df[col] == 0).sum()
        print(f"{col}: Valeurs manquantes = {missing}, Zéros = {zeros}, Min = {df[col].min()}, Max = {df[col].max()}, Moyenne = {df[col].mean():.2f}")
    else:
        print(f"{col}: colonne non trouvée")

# Corrélations rapides
print("\n=== Corrélations (variables clés) ===")
cols_to_corr = [c for c in critical_cols if c in df.columns]
if len(cols_to_corr) > 1:
    print(df[cols_to_corr].corr())
else:
    print("Pas assez de colonnes clés pour calculer les corrélations.")

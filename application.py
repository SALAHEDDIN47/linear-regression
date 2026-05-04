# ============================================
# APPLICATION COMPLÈTE DU CHAPITRE 6
# Détection et traitement de la non-linéarité
# Datasets : diamonds, anscombe, tips
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f as f_dist
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 70)
print("CHAPITRE 6 : DÉTECTION ET TRAITEMENT DE LA NON-LINÉARITÉ")
print("=" * 70)







print("\n" + "=" * 70)
print("PARTIE 1 : RÉGRESSION SIMPLE - DÉTECTION GRAPHIQUE")
print("Section 6.1.1 - Dataset : diamonds")
print("=" * 70)

# Chargement d'un échantillon (pour performances)
diamonds = sns.load_dataset("diamonds").sample(1000, random_state=42)

# Sélection des variables
X_dia = diamonds['carat']
Y_dia = diamonds['price']

print(f"Nombre d'observations : {len(diamonds)}")
print(f"Variable X : carat (poids en carats)")
print(f"Variable Y : price (prix en dollars)")

# === Graphique 1 : Nuage de points avec régression linéaire ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Données brutes + droite de régression
axes[0].scatter(X_dia, Y_dia, alpha=0.3, s=10)
axes[0].set_xlabel('Carat')
axes[0].set_ylabel('Price ($)')
axes[0].set_title('Données brutes : Price ~ Carat')

# Régression linéaire simple
X_const = sm.add_constant(X_dia)
model_lin = sm.OLS(Y_dia, X_const).fit()
X_sorted = np.sort(X_dia)
Y_pred = model_lin.predict(sm.add_constant(X_sorted))
axes[0].plot(X_sorted, Y_pred, 'r-', linewidth=2, label=f'R² = {model_lin.rsquared:.4f}')
axes[0].legend()

# 2. Transformation log-log
axes[1].scatter(np.log(X_dia), np.log(Y_dia), alpha=0.3, s=10)
axes[1].set_xlabel('ln(Carat)')
axes[1].set_ylabel('ln(Price)')
axes[1].set_title('Transformation log-log')

# Régression sur données transformées
X_log = sm.add_constant(np.log(X_dia))
model_log = sm.OLS(np.log(Y_dia), X_log).fit()
X_log_sorted = np.sort(np.log(X_dia))
Y_log_pred = model_log.predict(sm.add_constant(X_log_sorted))
axes[1].plot(X_log_sorted, Y_log_pred, 'r-', linewidth=2, label=f'R² = {model_log.rsquared:.4f}')
axes[1].legend()

# 3. Comparaison dans l'espace original
axes[2].scatter(X_dia, Y_dia, alpha=0.3, s=10, label='Données')
axes[2].plot(X_sorted, Y_pred, 'r-', linewidth=2, alpha=0.7, label='Modèle linéaire')
axes[2].plot(X_sorted, np.exp(model_log.predict(sm.add_constant(np.log(X_sorted)))),
             'g-', linewidth=2, label='Modèle log-log')
axes[2].set_xlabel('Carat')
axes[2].set_ylabel('Price ($)')
axes[2].set_title('Comparaison des modèles')
axes[2].legend()

plt.tight_layout()
plt.savefig('partie1_detection_graphique.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nR² linéaire simple : {model_lin.rsquared:.4f}")
print(f"R² log-log : {model_log.rsquared:.4f}")
print(f"Amélioration : ΔR² = {model_log.rsquared - model_lin.rsquared:.4f}")
print("→ La transformation log-log améliore nettement l'ajustement")




print("\n" + "=" * 70)
print("PARTIE 2 : η² vs r² + TEST DE LINÉARITÉ W²")
print("Section 6.1.2 - Dataset : anscombe (II)")
print("=" * 70)


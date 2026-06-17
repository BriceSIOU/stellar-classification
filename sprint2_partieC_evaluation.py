"""
================================================================================
SDSS17 - Sprint 2 : PARTIE C - Évaluation comparative ML (Version corrigée)
Auteur : TOBOSSI Sarah
Objectif : Comparaison Random Forest (Brice) vs XGBoost (Sarah T.)

PRÉREQUIS :
- Les modèles best_rf et best_xgb sont déjà entraînés dans la session
- Les variables suivantes existent : X_test_scaled, y_test, le (LabelEncoder)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("=" * 80)
print("SDSS17 - SPRINT 2 : PARTIE C - ÉVALUATION COMPARATIVE ML")
print("=" * 80)
print("Auteur : TOBOSSI Sarah")
print("Modèles : Random Forest (Brice) vs XGBoost (Sarah T.)")
print("=" * 80)

# ============================================================================
# 1. VÉRIFICATION DES MODÈLES DISPONIBLES
# ============================================================================
print("\n" + "=" * 60)
print("1. VÉRIFICATION DES MODÈLES")
print("=" * 60)

# Vérifier l'existence des variables nécessaires
required_vars = ['best_rf', 'best_xgb', 'X_test_scaled', 'y_test', 'le']
missing_vars = [var for var in required_vars if var not in dir()]

if missing_vars:
    print(f"❌ Variables manquantes : {missing_vars}")
    print("Assurez-vous d'avoir exécuté les codes de Brice et Sarah T. avant celui-ci")
    raise NameError(f"Variables manquantes : {missing_vars}")

print("✅ Toutes les variables nécessaires sont présentes")

# Récupération des modèles
rf_model = best_rf
xgb_model = best_xgb

# Vérification des classes
class_names = le.classes_
n_classes = len(class_names)
print(f"Classes : {class_names}")
print(f"Test set size : {len(y_test)} échantillons")

# ============================================================================
# 2. PRÉDICTIONS ET PROBABILITÉS
# ============================================================================
print("\n" + "=" * 60)
print("2. PRÉDICTIONS DES MODÈLES")
print("=" * 60)

# Prédictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Probabilités pour AUC-ROC
y_proba_rf = rf_model.predict_proba(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)

# Binarisation pour multi-classes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

print("✅ Prédictions calculées")

# ============================================================================
# 3. MÉTRIQUES DÉTAILLÉES
# ============================================================================
print("\n" + "=" * 60)
print("3. MÉTRIQUES DÉTAILLÉES")
print("=" * 60)

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# F1 scores
f1_macro_rf = f1_score(y_test, y_pred_rf, average='macro')
f1_macro_xgb = f1_score(y_test, y_pred_xgb, average='macro')

f1_weighted_rf = f1_score(y_test, y_pred_rf, average='weighted')
f1_weighted_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

f1_per_class_rf = f1_score(y_test, y_pred_rf, average=None)
f1_per_class_xgb = f1_score(y_test, y_pred_xgb, average=None)

# AUC-ROC (macro average)
auc_rf = roc_auc_score(y_test_bin, y_proba_rf, multi_class='ovr', average='macro')
auc_xgb = roc_auc_score(y_test_bin, y_proba_xgb, multi_class='ovr', average='macro')

# Tableau comparatif
print("\n" + "=" * 80)
print("TABLEAU COMPARATIF DES PERFORMANCES")
print("=" * 80)
print(f"{'Métrique':<20} {'Random Forest':<20} {'XGBoost':<20} {'Écart':<15}")
print("-" * 75)
print(f"{'Accuracy':<20} {acc_rf:<20.4f} {acc_xgb:<20.4f} {acc_xgb - acc_rf:+.4f}")
print(f"{'F1-score (macro)':<20} {f1_macro_rf:<20.4f} {f1_macro_xgb:<20.4f} {f1_macro_xgb - f1_macro_rf:+.4f}")
print(f"{'F1-score (weighted)':<20} {f1_weighted_rf:<20.4f} {f1_weighted_xgb:<20.4f} {f1_weighted_xgb - f1_weighted_rf:+.4f}")
print(f"{'AUC-ROC (macro)':<20} {auc_rf:<20.4f} {auc_xgb:<20.4f} {auc_xgb - auc_rf:+.4f}")

# F1 par classe
print("\n" + "=" * 80)
print("F1-SCORE PAR CLASSE")
print("=" * 80)
print(f"{'Classe':<15} {'Random Forest':<20} {'XGBoost':<20} {'Différence':<15}")
print("-" * 70)
for i, name in enumerate(class_names):
    diff = f1_per_class_xgb[i] - f1_per_class_rf[i]
    print(f"{name:<15} {f1_per_class_rf[i]:<20.4f} {f1_per_class_xgb[i]:<20.4f} {diff:+.4f}")

# ============================================================================
# 4. CLASSIFICATION REPORTS COMPLETS
# ============================================================================
print("\n" + "=" * 60)
print("4. CLASSIFICATION REPORTS")
print("=" * 60)

print("\n📊 RANDOM FOREST - Classification Report")
print("-" * 50)
print(classification_report(y_test, y_pred_rf, target_names=class_names, digits=4))

print("\n📊 XGBOOST - Classification Report")
print("-" * 50)
print(classification_report(y_test, y_pred_xgb, target_names=class_names, digits=4))

# ============================================================================
# 5. MATRICES DE CONFUSION (côte à côte)
# ============================================================================
print("\n" + "=" * 60)
print("5. GÉNÉRATION DES MATRICES DE CONFUSION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_names)
disp_rf.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Random Forest (tuned)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Prédiction', fontsize=11)
axes[0].set_ylabel('Vérité terrain', fontsize=11)

# XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=class_names)
disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('XGBoost (tuned)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Prédiction', fontsize=11)
axes[1].set_ylabel('Vérité terrain', fontsize=11)

plt.suptitle('Comparaison des Matrices de Confusion', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Matrices de confusion sauvegardées : 'confusion_matrices_comparison.png'")

# Analyse des erreurs
print("\n📊 Analyse des erreurs :")
fp_rf = cm_rf.sum(axis=0) - np.diag(cm_rf)
fn_rf = cm_rf.sum(axis=1) - np.diag(cm_rf)
fp_xgb = cm_xgb.sum(axis=0) - np.diag(cm_xgb)
fn_xgb = cm_xgb.sum(axis=1) - np.diag(cm_xgb)

print("\nRandom Forest :")
for i, name in enumerate(class_names):
    print(f"  {name} → Faux positifs: {fp_rf[i]}, Faux négatifs: {fn_rf[i]}")
print(f"  Total erreurs : {cm_rf.sum() - np.trace(cm_rf)}")

print("\nXGBoost :")
for i, name in enumerate(class_names):
    print(f"  {name} → Faux positifs: {fp_xgb[i]}, Faux négatifs: {fn_xgb[i]}")
print(f"  Total erreurs : {cm_xgb.sum() - np.trace(cm_xgb)}")

# ============================================================================
# 6. COURBES ROC MULTI-CLASSES (comparaison)
# ============================================================================
print("\n" + "=" * 60)
print("6. GÉNÉRATION DES COURBES ROC")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#4C72B0', '#55A868', '#DD8452']

# Random Forest
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_rf[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_proba_rf[:, i])
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_i:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
axes[0].set_xlabel('Taux de faux positifs (FPR)')
axes[0].set_ylabel('Taux de vrais positifs (TPR)')
axes[0].set_title('Random Forest - Courbes ROC', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# XGBoost
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_xgb[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_proba_xgb[:, i])
    axes[1].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_i:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
axes[1].set_xlabel('Taux de faux positifs (FPR)')
axes[1].set_ylabel('Taux de vrais positifs (TPR)')
axes[1].set_title('XGBoost - Courbes ROC', fontsize=13, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.suptitle('Comparaison des Courbes ROC (One-vs-Rest)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Courbes ROC sauvegardées : 'roc_curves_comparison.png'")

# ============================================================================
# 7. BARPLOT F1-SCORE PAR CLASSE
# ============================================================================
print("\n" + "=" * 60)
print("7. VISUALISATION F1-SCORE PAR CLASSE")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(class_names))
width = 0.35

bars1 = ax.bar(x - width/2, f1_per_class_rf, width, label='Random Forest', 
               color='steelblue', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, f1_per_class_xgb, width, label='XGBoost',
               color='forestgreen', edgecolor='black', linewidth=1)

# Ajouter les valeurs sur les barres
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Classe', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-score', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des F1-score par classe', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylim(0.92, 1.01)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('f1_comparison_by_class.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Comparaison F1 sauvegardée : 'f1_comparison_by_class.png'")

# ============================================================================
# 8. SYNTHÈSE DES FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 60)
print("8. SYNTHÈSE FEATURE IMPORTANCE")
print("=" * 60)

# Définition des noms de features (à adapter selon ton dataset)
feature_names = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
                 'redshift', 'u-g', 'g-r', 'r-i', 'i-z']

try:
    if hasattr(rf_model, 'feature_importances_'):
        rf_imp = rf_model.feature_importances_
        if len(rf_imp) == len(feature_names):
            rf_top_idx = np.argsort(rf_imp)[::-1][0]
            print(f"Random Forest - Feature la plus importante : {feature_names[rf_top_idx]} ({rf_imp[rf_top_idx]:.4f})")
            if 'redshift' in feature_names:
                redshift_idx = feature_names.index('redshift')
                print(f"  - redshift = {rf_imp[redshift_idx]:.4f} ({rf_imp[redshift_idx]*100:.1f}%)")
    else:
        print("Random Forest - Feature importance non disponible")
except Exception as e:
    print(f"Random Forest - Erreur récupération importance : {e}")

try:
    if hasattr(xgb_model, 'feature_importances_'):
        xgb_imp = xgb_model.feature_importances_
        if len(xgb_imp) == len(feature_names):
            xgb_top_idx = np.argsort(xgb_imp)[::-1][0]
            print(f"XGBoost - Feature la plus importante : {feature_names[xgb_top_idx]} ({xgb_imp[xgb_top_idx]:.4f})")
            if 'redshift' in feature_names:
                redshift_idx = feature_names.index('redshift')
                print(f"  - redshift = {xgb_imp[redshift_idx]:.4f} ({xgb_imp[redshift_idx]*100:.1f}%)")
    else:
        print("XGBoost - Feature importance non disponible")
except Exception as e:
    print(f"XGBoost - Erreur récupération importance : {e}")

# ============================================================================
# 9. SAUVEGARDE DES RÉSULTATS POUR SPRINT 4
# ============================================================================
print("\n" + "=" * 60)
print("9. SAUVEGARDE DES RÉSULTATS")
print("=" * 60)

# Dictionnaire complet pour le Sprint 4 (tableau comparatif)
ml_comparison_results = {
    'random_forest': {
        'accuracy': round(acc_rf, 4),
        'f1_macro': round(f1_macro_rf, 4),
        'f1_weighted': round(f1_weighted_rf, 4),
        'auc_roc': round(auc_rf, 4),
        'f1_per_class': {name: round(score, 4) for name, score in zip(class_names, f1_per_class_rf)}
    },
    'xgboost': {
        'accuracy': round(acc_xgb, 4),
        'f1_macro': round(f1_macro_xgb, 4),
        'f1_weighted': round(f1_weighted_xgb, 4),
        'auc_roc': round(auc_xgb, 4),
        'f1_per_class': {name: round(score, 4) for name, score in zip(class_names, f1_per_class_xgb)}
    },
    'comparison': {
        'best_accuracy': 'XGBoost' if acc_xgb > acc_rf else 'Random Forest',
        'best_f1_macro': 'XGBoost' if f1_macro_xgb > f1_macro_rf else 'Random Forest',
        'best_auc': 'XGBoost' if auc_xgb > auc_rf else 'Random Forest',
        'accuracy_gap': abs(acc_xgb - acc_rf),
        'f1_gap': abs(f1_macro_xgb - f1_macro_rf),
        'auc_gap': abs(auc_xgb - auc_rf)
    }
}

# Sauvegarde en JSON
with open('ml_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(ml_comparison_results, f, indent=2, ensure_ascii=False)

print("✅ Résultats sauvegardés dans 'ml_comparison_results.json'")

# Export CSV pour tableau
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [acc_rf, acc_xgb],
    'F1_macro': [f1_macro_rf, f1_macro_xgb],
    'F1_weighted': [f1_weighted_rf, f1_weighted_xgb],
    'AUC_ROC': [auc_rf, auc_xgb]
})
comparison_df.to_csv('ml_comparison_table.csv', index=False)
print("✅ Tableau CSV sauvegardé : 'ml_comparison_table.csv'")

# ============================================================================
# 10. RAPPORT FINAL POUR SPRINT 4
# ============================================================================
print("\n" + "=" * 80)
print("10. RAPPORT COMPARATIF - SPRINT 2 (PARTIE C)")
print("=" * 80)

# Construction du rapport (version corrigée sans erreur d'indentation)
report_lines = []
report_lines.append("=" * 80)
report_lines.append("         ANALYSE COMPARATIVE ML : RF (Brice) vs XGB (Sarah T.)")
report_lines.append("=" * 80)
report_lines.append(f"Auteur : TOBOSSI Sarah")
report_lines.append("")
report_lines.append("1. PERFORMANCES GLOBALES")
report_lines.append("   " + "-" * 60)
report_lines.append(f"   Accuracy         : RF = {acc_rf:.4f} | XGB = {acc_xgb:.4f} | Écart = {acc_xgb - acc_rf:+.4f}")
report_lines.append(f"   F1-score (macro) : RF = {f1_macro_rf:.4f} | XGB = {f1_macro_xgb:.4f} | Écart = {f1_macro_xgb - f1_macro_rf:+.4f}")
report_lines.append(f"   AUC-ROC (macro)  : RF = {auc_rf:.4f} | XGB = {auc_xgb:.4f} | Écart = {auc_xgb - auc_rf:+.4f}")
report_lines.append("")
report_lines.append("2. PERFORMANCES PAR CLASSE (F1-score)")
report_lines.append("   " + "-" * 60)

for i, name in enumerate(class_names):
    diff = f1_per_class_xgb[i] - f1_per_class_rf[i]
    report_lines.append(f"   {name:<12} : RF = {f1_per_class_rf[i]:.4f} | XGB = {f1_per_class_xgb[i]:.4f} | Diff = {diff:+.4f}")

report_lines.append("")
report_lines.append("3. ANALYSE DES ERREURS")
report_lines.append(f"   - Random Forest : {cm_rf.sum() - np.trace(cm_rf)} erreurs totales")
report_lines.append(f"   - XGBoost       : {cm_xgb.sum() - np.trace(cm_xgb)} erreurs totales")
report_lines.append("   - La confusion principale : GALAXY ↔ QSO (physiquement cohérent)")
report_lines.append("")
report_lines.append("4. CONCLUSION POUR SPRINT 3 (Deep Learning)")
report_lines.append(f"   ✅ Les deux modèles ML excellent (accuracy > {min(acc_rf, acc_xgb)*100:.1f}%)")
report_lines.append(f"   ⚠️ Le MLP devra dépasser {max(acc_rf, acc_xgb)*100:.2f}% d'accuracy")
report_lines.append("   ⚠️ L'amélioration du F1-score pour QSO est le principal défi")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("✅ SPRINT 2 - PARTIE C TERMINÉE")
report_lines.append("=" * 80)

report = "\n".join(report_lines)
print(report)

# Sauvegarde du rapport
with open('ml_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✅ Rapport sauvegardé : 'ml_analysis_report.txt'")
print("\n" + "=" * 80)
print("🏁 FIN DE LA PARTIE C - RÉSULTATS PRÊTS POUR SPRINT 4")
print("=" * 80)

# Affichage récapitulatif final
print("\n📁 Fichiers générés :")
print("   - confusion_matrices_comparison.png")
print("   - roc_curves_comparison.png")
print("   - f1_comparison_by_class.png")
print("   - ml_comparison_results.json")
print("   - ml_comparison_table.csv")
print("   - ml_analysis_report.txt")
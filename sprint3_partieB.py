"""
================================================================================
SDSS17 - SPRINT 3 : PARTIE B2 - Entraînement & Courbes d'apprentissage
Auteur : TOBOSSI Sarah
Objectif : Entraîner le MLP et visualiser les courbes d'apprentissage
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SPRINT 3 - PARTIE B2 : ENTRAÎNEMENT & COURBES D'APPRENTISSAGE")
print("=" * 80)
print("Auteur : TOBOSSI Sarah")
print("=" * 80)

# ============================================================================
# 1. VÉRIFICATION
# ============================================================================
print("\n1. VÉRIFICATION DES DONNÉES...")

if 'X_train_resampled' not in globals():
    raise SystemExit("❌ Données manquantes - Exécutez d'abord le Sprint 1")

if 'mlp' not in globals():
    raise SystemExit("❌ Architecture MLP non définie - Exécutez d'abord la Partie B1")

print(f"✅ Données prêtes : {X_train_resampled.shape[0]} échantillons")
print(f"✅ Architecture MLP déjà définie")

# ============================================================================
# 2. ENTRAÎNEMENT
# ============================================================================
print("\n2. ENTRAÎNEMENT DU MLP...")

n_features = X_train_resampled.shape[1]
n_classes = len(np.unique(y_train_resampled))

print(f"   Architecture : {n_features} → 128 → 64 → 32 → {n_classes}")
print(f"   Early stopping : patience=10, validation=10%")
print(f"   Début de l'entraînement...\n")

start_time = time.time()
mlp.fit(X_train_resampled, y_train_resampled)
training_time = time.time() - start_time

print(f"\n✅ Entraînement terminé en {training_time:.2f} secondes")
print(f"   - Nombre d'itérations : {mlp.n_iter_}")
print(f"   - Loss finale         : {mlp.loss_:.6f}")

# ============================================================================
# 3. COURBE D'APPRENTISSAGE - LOSS
# ============================================================================
print("\n3. COURBE D'APPRENTISSAGE - LOSS...")

plt.figure(figsize=(10, 5))

if hasattr(mlp, 'loss_curve_') and len(mlp.loss_curve_) > 0:
    plt.plot(mlp.loss_curve_, linewidth=2, color='blue')
    plt.xlabel('Itérations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Courbe d\'apprentissage - Loss (MLP)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Point final
    final_loss = mlp.loss_curve_[-1]
    plt.scatter(len(mlp.loss_curve_)-1, final_loss, color='red', s=100, zorder=5)
    plt.annotate(f'Loss finale: {final_loss:.6f}',
                 xy=(len(mlp.loss_curve_)-1, final_loss),
                 xytext=(len(mlp.loss_curve_)-25, final_loss + 0.02),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=150)
    plt.show()
    print("✓ Courbe de loss sauvegardée : 'mlp_loss_curve.png'")
    
    # Sauvegarde CSV
    np.savetxt('mlp_loss_curve.csv', mlp.loss_curve_, delimiter=',', header='loss', comments='')
    print("✓ Loss curve sauvegardée : 'mlp_loss_curve.csv'")

# ============================================================================
# 4. COURBE D'APPRENTISSAGE - VALIDATION
# ============================================================================
print("\n4. COURBE D'APPRENTISSAGE - VALIDATION...")

if hasattr(mlp, 'validation_scores_') and len(mlp.validation_scores_) > 0:
    plt.figure(figsize=(10, 5))
    
    plt.plot(mlp.validation_scores_, linewidth=2, color='green')
    plt.xlabel('Itérations', fontsize=12)
    plt.ylabel('Accuracy validation', fontsize=12)
    plt.title('Courbe d\'apprentissage - Accuracy Validation (MLP)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Meilleur score
    best_score = max(mlp.validation_scores_)
    best_iter = np.argmax(mlp.validation_scores_)
    plt.scatter(best_iter, best_score, color='red', s=100, zorder=5)
    plt.annotate(f'Best: {best_score:.4f}',
                 xy=(best_iter, best_score),
                 xytext=(best_iter + 5, best_score - 0.02),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Ligne du meilleur score
    plt.axhline(y=best_score, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_score:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_validation_curve.png', dpi=150)
    plt.show()
    print("✓ Courbe de validation sauvegardée : 'mlp_validation_curve.png'")
    
    # Sauvegarde CSV
    np.savetxt('mlp_validation_scores.csv', mlp.validation_scores_, delimiter=',',
               header='validation_accuracy', comments='')
    print("✓ Validation scores sauvegardés : 'mlp_validation_scores.csv'")
    
    print(f"\n📈 Meilleur score de validation : {best_score:.4f} (itération {best_iter + 1})")

# ============================================================================
# 5. SAUVEGARDE DU MODÈLE
# ============================================================================
print("\n5. SAUVEGARDE DU MODÈLE...")

joblib.dump(mlp, 'mlp_model_sprint3.pkl')
print("✅ Modèle sauvegardé : 'mlp_model_sprint3.pkl'")

# ============================================================================
# 6. RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("RÉSUMÉ - SPRINT 3 PARTIE B2")
print("=" * 80)

print(f"""
📊 RÉSULTATS DE L'ENTRAÎNEMENT :

   Architecture : 12 → 128 → 64 → 32 → 3
   Temps        : {training_time:.2f} secondes
   Itérations   : {mlp.n_iter_}
   Loss finale  : {mlp.loss_:.6f}
   """)

if hasattr(mlp, 'validation_scores_'):
    print(f"   Best validation : {max(mlp.validation_scores_):.4f}")

print("""
📁 FICHIERS GÉNÉRÉS :
   ├── mlp_model_sprint3.pkl       (Modèle)
   ├── mlp_loss_curve.png          (Courbe loss)
   ├── mlp_loss_curve.csv          (Loss values)
   ├── mlp_validation_curve.png    (Courbe validation)
   └── mlp_validation_scores.csv   (Validation values)

✅ SPRINT 3 - PARTIE B2 TERMINÉE
   → Passer à la Partie C (Évaluation)
""")

print("=" * 80)
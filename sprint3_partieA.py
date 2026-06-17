"""
================================================================================
SDSS17 - SPRINT 3 : PARTIE B1 - Architecture MLP & Dropout
Auteur : TOBOSSI Sarah
Objectif : Définir l'architecture du MLP avec Dropout
================================================================================
"""

from sklearn.neural_network import MLPClassifier

print("=" * 80)
print("SPRINT 3 - PARTIE B1 : ARCHITECTURE MLP & DROPOUT")
print("=" * 80)
print("Auteur : TOBOSSI Sarah")
print("=" * 80)

# ============================================================================
# ARCHITECTURE MLP AVEC DROPOUT (via régularisation L2)
# ============================================================================

mlp = MLPClassifier(
    # Architecture
    hidden_layer_sizes=(128, 64, 32),    # 3 couches cachées
    activation='relu',                    # Fonction d'activation ReLU
    
    # Optimiseur
    solver='adam',                        # Adam
    learning_rate_init=0.001,             # Taux d'apprentissage
    learning_rate='adaptive',             # Adaptatif
    
    # DROPOUT (via régularisation L2)
    alpha=0.0001,                         # Équivalent au dropout
    
    # Entraînement
    batch_size=32,
    max_iter=100,
    shuffle=True,
    
    # Early stopping
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    
    random_state=42,
    verbose=False
)

print("\n📐 ARCHITECTURE MLP :")
print("   ┌─────────────────────────────────────────────────────────┐")
print("   │  Couche d'entrée    : 12 neurones (features)            │")
print("   │  ↓                                                      │")
print("   │  Couche Dense 1     : 128 neurones (ReLU)               │")
print("   │  Dropout (via L2)   : alpha = 0.0001                    │")
print("   │  ↓                                                      │")
print("   │  Couche Dense 2     : 64 neurones (ReLU)                │")
print("   │  Dropout (via L2)   : alpha = 0.0001                    │")
print("   │  ↓                                                      │")
print("   │  Couche Dense 3     : 32 neurones (ReLU)                │")
print("   │  ↓                                                      │")
print("   │  Couche de sortie   : 3 neurones (Softmax)              │")
print("   │                      (GALAXY, QSO, STAR)                │")
print("   └─────────────────────────────────────────────────────────┘")

print("\n📊 PARAMÈTRES :")
print(f"   - Optimiseur        : Adam (learning_rate = 0.001)")
print(f"   - Batch size        : 32")
print(f"   - Max iterations    : 100")
print(f"   - Early stopping    : Oui (patience = 10)")
print(f"   - Validation split  : 10%")
print(f"   - DROPOUT (L2)      : alpha = 0.0001")

print("\n✅ Architecture MLP définie")
print("   → Passer à la Partie B2 (Entraînement & Courbes)")

print("=" * 80)
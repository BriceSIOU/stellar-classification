### 1. Présentation du projet

Problématique : Comment classifier automatiquement un objet céleste (étoile, galaxie, quasar) à partir de ses mesures photométriques et spectrales ?

Objectif : Construire et comparer des modèles ML et DL capables de prédire la classe d'un objet céleste avec la meilleure précision possible.
Type du problème : Classification multi-classe supervisée (3 classes : STAR, GALAXY, QSO)

Preprocessing → suppression des colonnes inutiles (IDs), gestion des valeurs manquantes, SMOTE pour le déséquilibre
Feature Engineering → création des différences de couleur u-g, g-r, r-i, i-z
Normalisation → standardisation (moyenne 0, écart-type 1)
Modèles ML → Random Forest, XGBoost
Modèle DL → MLP (Multilayer Perceptron)
Évaluation → F1 par classe, matrice de confusion, AUC-ROC
Comparaison ML vs DL → tableau récapitulatif des performances des 3 modèles

## https://www.researchgate.net/publication/398787690_Star_Classification_Using_Machine_Learning_A_Comparative_Analysis_of_Random_Forest_and_LightGBM_on_SDSS_Data
# KNN Vanilla

Implémentation de l'algorithme K-Nearest Neighbors (KNN) en orienté objet, en pure Python, sans aucune librairie externe pour les calculs.

---

## Contrainte

Le seul module autorisé est `data_loader.py` pour le chargement et la normalisation des données.

---

## Structure de la classe KNN

### Attributs

- `k` — nombre de voisins à considérer
- `X_train` — données d'entraînement
- `y_train` — labels d'entraînement

### Méthodes

#### `fit(X, y)`

Stocke les données d'entraînement. Le KNN est un algorithme "lazy" : il ne calcule rien à l'entraînement, tout se passe à la prédiction.

#### `euclidean_distance(point_a, point_b)`

Calcule la distance euclidienne entre deux points en pure Python :

- Pour chaque dimension, on calcule la différence entre les deux points
- On l'élève au carré et on additionne
- On retourne la racine carrée du total

#### `predict(X)`

Pour chaque point à prédire :

1. Calcule la distance avec chaque point d'entraînement
2. Trie les distances et garde les `k` plus proches voisins
3. Retourne la classe majoritaire parmi ces voisins (vote)

#### `evaluate(X, y)`

Appelle `predict`, compare les prédictions aux vrais labels et retourne l'**accuracy** :

```
accuracy = nombre de bonnes prédictions / nombre total de prédictions
```

Le F1 score a été écarté car son implémentation en pure Python (sans librairie) est trop complexe dans ce contexte pour moi

---

## Utilisation

```python
from data_loader import load_normalized_data
from knn import KNN

X, y, scaler = load_normalized_data(file_path="bienetre.xlsx", target_col="target")

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

model = KNN(k=3)
model.fit(X_train, y_train)

score = model.evaluate(X_test, y_test)
print("Accuracy :", score)
```

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

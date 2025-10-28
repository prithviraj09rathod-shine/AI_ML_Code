import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class SGDLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_weights(n)

        for epoch in range(self.epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)

            for i in range(0, m, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                y_pred = self.predict_proba(X_batch)
                error = y_pred - y_batch

                dw = np.dot(X_batch.T, error) / len(batch_idx)
                db = np.sum(error) / len(batch_idx)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            if epoch % 10 == 0:
                loss = self.compute_loss(y, self.predict_proba(X))
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        print("Training complete.")

X = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [2.0, 2.0], [2.5, 1.5]])
#w = np.random.randn(X.shape[1])
y = np.array([0, 0, 0, 1, 1])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Training the model
model = SGDLogisticRegression(learning_rate=0.1, epochs=100, batch_size=2)
model.fit(X, y)
#Predicting the labels
predictions = model.predict(X)
print("Predicted labels:", predictions)

# Evaluate
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

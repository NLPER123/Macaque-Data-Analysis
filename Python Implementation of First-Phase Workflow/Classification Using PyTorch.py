import torch
import torch.nn as nn
import torch.optim as optim

class NearestMedianClassifier:
    def __init__(self):
        self.medians = None
        self.labels = None

    def fit(self, X, y):
        """Train the nearest-median classifier."""
        classes = np.unique(y)
        medians = []
        for cls in classes:
            cls_data = X[y == cls]
            medians.append(np.median(cls_data, axis=0))
        self.medians = np.array(medians)
        self.labels = classes

    def predict(self, X):
        """Predict the class for each sample."""
        distances = np.linalg.norm(X[:, None] - self.medians[None, :], axis=2)
        return self.labels[np.argmin(distances, axis=1)]

    def score(self, X, y):
        """Compute the accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

import numpy as np

def generate_classification_data(n_samples=100, n_features=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    flip = np.random.rand(len(y)) < 0.05
    y[flip] = 1 - y[flip]
    return X, y

if __name__ == "__main__":
    X, y = generate_classification_data()
    print("Sample features (X):")
    print(X[:5])
    print("Sample labels (y):")
    print(y[:5])

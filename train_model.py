import numpy as np
from model.gradient_boosting import GradientBoostingClassifier
from generate_data import generate_classification_data

if __name__ == "__main__":
    X, y = generate_classification_data()
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    print(f"Training accuracy: {acc:.2f}")


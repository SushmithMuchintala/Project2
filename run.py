import argparse 
import numpy as np 
from generate_data import generate_classification_data 
from model.gradient_boosting import GradientBoostingClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    X, y = generate_classification_data(n_samples=2000)
    clf = GradientBoostingClassifier(n_estimators=args.n_estimators,
                                     learning_rate=args.lr,
                                     max_depth=args.depth,
                                     subsample=0.8,
                                     early_stopping_rounds=70)
    clf.fit(X, y)
    print("Train accuracy:", np.mean(clf.predict(X) == y))
    clf.plot_loss()

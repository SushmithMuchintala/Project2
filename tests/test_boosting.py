import numpy as np
from model.gradient_boosting import GradientBoostingClassifier


def make_nonlinear(n: int = 500, seed: int = 0):
    """Return a 2‑D nonlinear dataset (circle‑like) with equal class split."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    # Non‑linear boundary: circle of radius ≈1
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    return X, y


# 1. High‑accuracy test on nonlinear data (same as before)

def test_high_accuracy():
    X, y = make_nonlinear()
    clf = GradientBoostingClassifier(n_estimators=200,
                                     learning_rate=0.1,
                                     max_depth=3,
                                     subsample=0.8,
                                     early_stopping_rounds=20,
                                     val_fraction=0.0)
    clf.fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.90  # should fit fairly well


# 2. Edge‑case: single feature, step function (same as before but val_fraction=0)


def test_edge_case_single_feature():
    X = np.arange(10).reshape(-1, 1)
    y = (X.ravel() > 4).astype(int)
    clf = GradientBoostingClassifier(n_estimators=50,
                                     learning_rate=0.2,
                                     max_depth=2,
                                     val_fraction=0.0)
    clf.fit(X, y)
    assert (clf.predict(X) == y).all()



# 3. Predict‑probabilities shape & range

def test_predict_proba_shape_and_range():
    X, y = make_nonlinear(seed=1)
    clf = GradientBoostingClassifier(n_estimators=100,
                                     learning_rate=0.1,
                                     max_depth=2,
                                     val_fraction=0.0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))



# 4. Robustness to noisy labels (accuracy should not be perfect)

def test_noisy_labels_accuracy_lower():
    X, y = make_nonlinear(seed=2)
    rng = np.random.default_rng(2)
    flip = rng.random(len(y)) < 0.15 
    y_noisy = y.copy()
    y_noisy[flip] = 1 - y_noisy[flip]

    clf = GradientBoostingClassifier(n_estimators=200,
                                     learning_rate=0.1,
                                     max_depth=3,
                                     val_fraction=0.0)
    clf.fit(X, y_noisy)
    acc = (clf.predict(X) == y_noisy).mean()
    assert 0.80 < acc < 0.98

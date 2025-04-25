from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. Tiny tree building blocks
# ============================================================================

class _TreeLeaf:        
    def __init__(self, depth: int = 0) -> None:
        self.depth = depth
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _TreeLeaf | None = None
        self.right: _TreeLeaf | None = None
        self.gamma: float = 0.0  


class DecisionTree:       
    """
    Builds a depth-limited regression tree using gain from
    second-order Taylor expansion of logistic deviance.
    """

    def __init__(self,
                 max_depth: int = 3,
                 min_leaf: int = 5) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root: _TreeLeaf | None = None

    # --------- public -------------------------------------------------------
    def fit(self,
            X: np.ndarray,
            grad: np.ndarray,
            hess: np.ndarray) -> "_TreeLeaf":
        """Grow the tree on gradients/Hessians."""
        self.root = self._build(X, grad, hess, depth=0)
        return self.root

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return leaf values (Newton updates) for each sample."""
        return np.array([self._traverse(row, self.root) for row in X])

    # --------- private helpers ---------------------------------------------
    def _build(self,
               X: np.ndarray,
               grad: np.ndarray,
               hess: np.ndarray,
               depth: int) -> _TreeLeaf:
        node = _TreeLeaf(depth)

      
        if depth >= self.max_depth or X.shape[0] <= self.min_leaf:
            node.gamma = self._leaf_value(grad, hess)
            return node

        best_gain = -np.inf
        best_feat, best_thr = None, None

        G_total, H_total = grad.sum(), hess.sum()

        for j in range(X.shape[1]):
            for thr in self._candidate_thresholds(X[:, j]):
                idx_left = X[:, j] <= thr
                if idx_left.sum() < self.min_leaf or (~idx_left).sum() < self.min_leaf:
                    continue  

                Gl, Hl = grad[idx_left].sum(), hess[idx_left].sum()
                Gr, Hr = G_total - Gl, H_total - Hl

                gain = 0.5 * ((Gl**2) / (Hl + 1e-12) +
                              (Gr**2) / (Hr + 1e-12))
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, j, thr

        if best_feat is None:
            node.gamma = self._leaf_value(grad, hess)
            return node

        node.feature, node.threshold = best_feat, best_thr
        mask = X[:, best_feat] <= best_thr
        node.left = self._build(X[mask], grad[mask], hess[mask], depth + 1)
        node.right = self._build(X[~mask], grad[~mask], hess[~mask], depth + 1)
        return node

    # -----------------------------------------------------------------------
    @staticmethod
    def _leaf_value(grad: np.ndarray, hess: np.ndarray) -> float:
        """Newton step for logistic loss (one leaf)."""
        return grad.sum() / (hess.sum() + 1e-12)

    @staticmethod
    def _candidate_thresholds(col: np.ndarray) -> np.ndarray:
        """Mid-points between unique sorted values => fewer splits to test."""
        uniq = np.unique(col)
        if uniq.size <= 1:
            return np.array([])
        return (uniq[:-1] + uniq[1:]) * 0.5

    def _traverse(self, row: np.ndarray, node: _TreeLeaf) -> float:
        """Follow the row down the tree until a leaf is hit."""
        if node.left is None:           
            return node.gamma
        if row[node.feature] <= node.threshold: 
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

# ============================================================================
# 2. Gradient-Boosting wrapper
# ============================================================================

class GradientBoostingClassifier:
    """
    Binary classifier that boosts shallow CARTs.
    Parameters mirror popular libraries (n_estimators, learning_rate, …)
    but the implementation is 100 % NumPy.
    """


    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 subsample: float = 1.0,
                 colsample: float = 1.0,
                 early_stopping_rounds: int | None = None,
                 val_fraction: float = 0.1,
                 random_state: int = 42) -> None:

        self.n_estimators = n_estimators
        self.shrink = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.rng = np.random.default_rng(random_state)

      
        self.trees: list[tuple[DecisionTree, np.ndarray]] = []
        self.losses: list[float] = []
        self.val_losses: list[float] = []

    # ----- training ---------------------------------------------------------
    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> "GradientBoostingClassifier":
        """Train the ensemble; y must be 0/1."""
        X, y = X.copy(), y.astype(float)

        X, y, X_val, y_val = self._train_val_split(X, y)

        raw_score = np.zeros(len(y))
        best_val, best_iter = np.inf, -1

        for m in range(self.n_estimators):
            # row & column sampling
            row_idx = (self.rng.choice(len(y),
                                       int(len(y) * self.subsample),
                                       replace=False)
                        if self.subsample < 1.0 else
                        np.arange(len(y)))

            col_idx = (self.rng.choice(X.shape[1],
                                       int(X.shape[1] * self.colsample),
                                       replace=False)
                        if self.colsample < 1.0 else
                        np.arange(X.shape[1]))

            grad, hess = self._grad_hess(y[row_idx], raw_score[row_idx])

            tree = DecisionTree(self.max_depth)
            tree.fit(X[row_idx][:, col_idx], grad, hess)

       
            update = np.zeros_like(raw_score)
            update[row_idx] = tree.predict(X[row_idx][:, col_idx])
            raw_score += self.shrink * update
            self.trees.append((tree, col_idx))

         
            train_loss = self._log_loss(y, raw_score)
            self.losses.append(train_loss)
            info = f"[{m+1}] train-logloss={train_loss:.4f}"

            if X_val is not None:
                val_loss = self._log_loss(y_val, self._raw_predict(X_val))
                self.val_losses.append(val_loss)
                info += f"  val-logloss={val_loss:.4f}"

             
                if val_loss < best_val - 1e-6:
                    best_val, best_iter = val_loss, m
                elif (self.early_stopping_rounds is not None and
                      m - best_iter >= self.early_stopping_rounds):
                    print(f"Early stop @ {m+1}")
                    break
            print(info)  

        return self

    # ----- inference --------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-2 * self._raw_predict(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ----- plotting ---------------------------------------------------------
    def plot_loss(self, title: str = "GBDT log-loss curve") -> None:
        plt.figure(figsize=(8, 5))                 
        plt.plot(self.losses, label="train")
        if self.val_losses:
            plt.plot(self.val_losses, label="val")
        plt.title(title or "Gradient-Boosting Log-Loss vs Iteration",
                  fontsize=14, weight="bold", pad=10)
        plt.xlabel("Iteration")
        plt.ylabel("Log-loss")
        plt.legend(); plt.grid(); plt.show()

    # ----- internal helpers -------------------------------------------------
    @staticmethod
    def _grad_hess(y: np.ndarray,
                   raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First/second derivatives of logistic deviance."""
        prob = 1.0 / (1.0 + np.exp(-2 * raw))
        grad = 2 * (y - prob)
        hess = 4 * prob * (1 - prob)
        return grad, hess

    @staticmethod
    def _log_loss(y: np.ndarray, raw: np.ndarray) -> float:
        return np.mean(np.log(1 + np.exp(-2 * y * raw)))

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        score = np.zeros(X.shape[0])
        for tree, cols in self.trees:
            score += self.shrink * tree.predict(X[:, cols])
        return score

    def _train_val_split(self, X, y):
        """Simple RNG shuffle split; returns X_train, y_train, X_val, y_val."""
        if not self.val_fraction:
            return X, y, None, None
        idx = self.rng.permutation(len(y))
        k = int(len(y) * self.val_fraction)
        return X[idx[k:]], y[idx[k:]], X[idx[:k]], y[idx[:k]]

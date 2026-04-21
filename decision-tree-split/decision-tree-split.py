import numpy as np

def decision_tree_split(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    def gini(labels):
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    best_feature = 0
    best_threshold = 0.0
    best_score = float("inf")

    for j in range(n_features):
        values = np.unique(X[:, j])

        
        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2.0

        for t in thresholds:
            left_mask = X[:, j] <= t
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if y_left.size == 0 or y_right.size == 0:
                continue

            g = (y_left.size / n_samples) * gini(y_left) + \
                (y_right.size / n_samples) * gini(y_right)

            if g < best_score:
                best_score = g
                best_feature = j
                best_threshold = float(t)

    return best_feature, best_threshold
import numpy as np

def gini_impurity(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    def gini(y):
        if y.size == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    n_left = y_left.size
    n_right = y_right.size
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    g_left = gini(y_left)
    g_right = gini(y_right)

    return (n_left / n_total) * g_left + (n_right / n_total) * g_right
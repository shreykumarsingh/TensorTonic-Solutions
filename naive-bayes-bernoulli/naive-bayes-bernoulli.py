import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train, y_train, X_test = map(np.array, (X_train, y_train, X_test))
    classes = np.sort(np.unique(y_train))
    res = []

    for x in X_test:
        row = []
        for c in classes:
            Xc = X_train[y_train == c]
            prior = np.log(len(Xc) / len(X_train))

            theta = (Xc.sum(axis=0) + 1) / (len(Xc) + 2)

            ll = np.sum(x * np.log(theta) +
                        (1 - x) * np.log(1 - theta))

            row.append(prior + ll)
        res.append(row)

    return np.array(res)
    pass
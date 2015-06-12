from cs231n.classifiers.linear_classifier import LinearSVM
X_train_feats = None
y_train = None
import numpy as np
y_val = None
X_val_feats = None
results = None
best_val = None

learning_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
regularization_strengths = [5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4]

for lr in learning_rates:
    for rs in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats, y_train, learning_rate=lr, reg=rs,
                  num_iters=3000, verbose=True)
        ac_train = np.mean(y_train == svm.predict(X_train_feats))
        ac_val = np.mean(y_val == svm.predict(X_val_feats))
        if ac_val > best_val:
            best_val = ac_val
            best_svm = svm
        key = (lr, rs)
        results[key] = ac_train, ac_val
from sklearn.svm import SVC


def run(X_train, y_train_cv):
    svm = SVC(C=10, kernel='rbf')
    svm.fit(X_train, y_train_cv.astype('int').values.ravel())
    return svm
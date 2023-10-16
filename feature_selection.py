from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest


def select_features_percentile(percentage, X_train, X_test, y_train):
    print("Selecting features with percentile...")
    percentile = percentage  # Select top features
    feature_selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)

    return [X_train_selected, X_test_selected]


def select_features_kbest(num_features, X_train, X_test, y_train):
    print("Selecting features with KBest...")
    feature_selector = SelectKBest(score_func=f_classif, k=num_features)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)

    return [X_train_selected, X_test_selected]


def perform_features_selection(fea, p, num_features, X_train_cv, X_test_cv, y_train_cv):
    if fea == 'kbest':
        (X_train_selected, X_test_selected) = select_features_kbest(num_features, X_train_cv,
                                                                    X_test_cv, y_train_cv)
    elif fea == 'percentile':
        (X_train_selected, X_test_selected) = select_features_percentile(p, X_train_cv,
                                                                         X_test_cv, y_train_cv)
    return [X_train_selected, X_test_selected]
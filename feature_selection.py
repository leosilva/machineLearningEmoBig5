from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
# from matplotlib import pyplot


def select_features_percentile(percentage, X_train, X_test, y_train):
    print("Selecting features with percentile...")
    percentile = percentage  # Select top features
    feature_selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)

    return [X_train_selected, X_test_selected]


def select_features_kbest(num_features, X_train, X_test, y_train, feature_to_include):
    print("Selecting features with KBest...")

    # columns_to_remove.remove(feature_to_include)

    columns_to_remove = [
        'o_score',
        'c_score',
        'e_score',
        'a_score',
        'n_score'
    ]

    columns_to_preserve = [
            # 'o_score',
            # 'c_score',
            # 'e_score',
            'a_score',
            # 'n_score'
        ]

    new_num_features = (num_features - len(columns_to_remove))

    new_columns = [c for c in X_train.columns if c not in columns_to_remove]

    # print(new_num_features)

    X_train_temp = X_train[new_columns]
    X_test_temp = X_test[new_columns]
    # print(X_train_temp.columns)

    feature_selector = SelectKBest(score_func=f_classif, k=new_num_features)
    # feature_selector = SelectKBest(score_func=f_classif, k=num_features)
    X_train_selected = feature_selector.fit_transform(X_train_temp, y_train)
    X_test_selected = feature_selector.transform(X_test_temp)

    # X_train_selected = feature_selector.fit_transform(X_train, y_train)
    # X_test_selected = feature_selector.transform(X_test)

    # print("X_train_selected shape: ", X_train_selected.shape)
    # print("X_test_selected shape: ", X_test_selected.shape)

    selected_indices = feature_selector.get_support(indices=True)
    selected_columns = X_train_temp.columns[selected_indices].tolist()
    selected_columns += columns_to_preserve

    # print("selected features: ", selected_columns)

    X_train_result = X_train[selected_columns]
    X_test_result = X_test[selected_columns]

    # print("X_train_result shape: ", X_train_result.shape)
    # print("X_test_result shape: ", X_test_result.shape)
    #
    # print("result features: ", + X_train_result.columns)

    return [X_train_result, X_test_result]
    # return [X_train_selected, X_test_selected]


def perform_features_selection(fea, num_features, X_train_cv, X_test_cv, y_train_cv, feature_to_include):
    if fea == 'kbest':
        (X_train_selected, X_test_selected) = select_features_kbest(num_features, X_train_cv, X_test_cv, y_train_cv, feature_to_include)
    elif fea == 'percentile':
        (X_train_selected, X_test_selected) = select_features_percentile(X_train_cv,
                                                                         X_test_cv, y_train_cv)
    return [X_train_selected, X_test_selected]



def permutation_feature_importance(model, X, y):
    result = {}
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
        fea = X.columns[i]
        result[fea] = round(v, 4)
    # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()
    return result
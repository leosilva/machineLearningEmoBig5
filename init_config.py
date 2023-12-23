import sklearn.metrics as me

def get_general_parameters(is_test):
    if is_test == 'True':
        print("Executing in TEST mode...")
        return {
            # "ngram": [(1,2)],
            "balance": [''],
            "percentage_features": [
                # 50,
                # 60,
                # 70,
                # 80,
                100
            ],
            "feature_selection": ['kbest'],
            "folds": [10],
            "feature_to_include": [
                'o_score', # nao considera esta feature, eh somente para entrar na funcao
                # 'c_score',
                # 'e_score',
                # 'a_score',
                # 'n_score'
            ]
        }
    else:
        print("Executing in PROD mode...")
        return {
            "ngram": [(1,2)],
            "balance": ['over', 'under', 'mixed', 'smote'],
            "percentage_features": [5, 10], #, 40, 50, 60, 70, 80, 90, 95],
            "feature_selection": ['kbest'], #, 'percentile'],
            "folds": [5, 10]
        }


def get_result_map():
    result_map = {
        "Algorithm": [],
        "Train Accuracy": [],
        "Train Precision": [],
        "Train Recall": [],
        "Train F1 Score": [],
        "Test Accuracy": [],
        "Test Precision": [],
        "Test Recall": [],
        "Test F1 Score": [],
        "Val. Accuracy": [],
        "Val. Precision": [],
        "Val. Recall": [],
        "Val. F1 Score": [],
        "AUC": [],
        # "Ngram": [],
        # "Vect. Strategy": [],
        "Bal. Strategy": [],
        "% of Features": [],
        "Folds": [],
        "Feat. Selec. Strategy": [],
        "Features": [],
        "Features Importance": [],
        # "Hyper Params.": [],
        "Model": []
    }
    return result_map


def get_default_scoring():
    scoring = {
        'roc_auc': me.make_scorer(me.roc_auc_score, needs_proba=True, multi_class='ovr'),
        'accuracy': me.make_scorer(me.accuracy_score),
        'precision': me.make_scorer(me.precision_score, average='weighted'),
        'recall': me.make_scorer(me.recall_score, average='weighted'),
        'f1_score': me.make_scorer(me.f1_score, average='macro'),
    }
    return scoring

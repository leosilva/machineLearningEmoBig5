from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import utils as ut
import numpy as np


RANDOM_STATE = ut.get_random_state()

# implementar outros classifiers, como SGDClassifier e GradientBoostingClassifier
def get_models(which_models):
    result = {}

    models = {
        'svc': {
            SVC(): {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                # 'gamma': ['auto'],
                # 'class_weight': ['balanced'], # se a opcao de execucao for -b True, este parametro eh removido
                'probability': [True],
                'random_state': [RANDOM_STATE]
            }
        },
        'random-forest': {
            RandomForestClassifier(): {
                'max_depth': [5, 10, 20, 50, 100, None],
                'n_estimators':  [500, 1000, 1500],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                # 'criterion': ['gini', 'entropy'],
                # 'class_weight':[None, 'balanced', 'balanced_subsample'],
                'random_state': [RANDOM_STATE]
            }
        },
        'multinomial-nb': {
            MultinomialNB(fit_prior=True): {
                'alpha': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                # "gamma": ['auto'],
                # "C": [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10, 100, 200, 1000, 1500, 1700, 2000],
                # 'random_state': [RANDOM_STATE]
            }
        },
        'logistic-regression': {
            LogisticRegression(max_iter=100000, multi_class='multinomial'): {
                'solver': ['lbfgs'],
                'penalty': [None, 'l2'],
                'C': [0.1, 1, 10],
                'random_state': [RANDOM_STATE]
            }
        },
        'decision-tree': {
            DecisionTreeClassifier() : {
                'max_depth': [3, 5, 10, 20, 50, 100],
                # 'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                # 'class_weight': [None, 'balanced'],
                'random_state': [RANDOM_STATE]
            }
        },
        'mlp-classifier': {
            MLPClassifier() : {
                'solver': ['lbfgs', 'sgd', 'adam'],
                'max_iter': [1000,1500,2000],
                'alpha': 10.0 ** -np.arange(1, 10),
                'hidden_layer_sizes':np.arange(10, 15),
                'random_state':[RANDOM_STATE]
            }
        }
    }

    for m in which_models:
        result.update(models[m])

    return result
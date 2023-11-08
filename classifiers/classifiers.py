from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import utils as ut
import numpy as np


# implementar outros classifiers, como SGDClassifier e GradientBoostingClassifier
def get_models(which_models):
    result = {}

    models = {
        'svc': {
            SVC(probability=True): {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                # 'gamma': ['auto'],
                # 'class_weight': ['balanced'], # se a opcao de execucao for -b True, este parametro eh removido
                'probability': [True]
            }
        },
        'random-forest': {
            RandomForestClassifier(): {
                # 'max_depth': [5, 10, 20, 50, 100, None],
                'n_estimators':  [5, 20, 50, 100],
                # 'min_samples_leaf': [5, 10, 20, 50, 100],
                # 'criterion': ['gini', 'entropy'],
                # 'class_weight':[None, 'balanced', 'balanced_subsample']
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
        'bernoulli-nb': {
            BernoulliNB(): {
                'alpha': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                # "gamma": ['auto'],
                # "C": [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10, 100, 200, 1000, 1500, 1700, 2000],
                # 'random_state': [RANDOM_STATE]
            }
        },
        'gaussian-nb': {
            GaussianNB(): {
                'var_smoothing': np.logspace(0, -9, num=100),
                # "gamma": ['auto'],
                # "C": [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 10, 100, 200, 1000, 1500, 1700, 2000],
                # 'random_state': [RANDOM_STATE]
            }
        },
        'complement-nb': {
            ComplementNB(fit_prior=True): {
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
                'C': [0.1, 1, 10]
            }
        },
        'decision-tree': {
            DecisionTreeClassifier() : {
                'max_depth': [3, 5, 10, 20, 50, 100],
                # 'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                # 'class_weight': [None, 'balanced']
            }
        },
        'mlp-classifier': {
            MLPClassifier() : {
                'solver': ['adam'],
                'max_iter': [100000],
                # 'alpha': 10.0 ** -np.arange(1, 10),
                # 'hidden_layer_sizes':np.arange(10, 15)
            }
        }
    }

    for m in which_models:
        result.update(models[m])

    return result
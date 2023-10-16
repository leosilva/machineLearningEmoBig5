from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def balance_classes_oversampling(X, y):
    print("Balancing classes with over sampling...")
    rus = RandomOverSampler()
    X_over, y_over = rus.fit_resample(X, y)

    return [X_over, y_over]


def balance_classes_undersampling(X, y):
    print("Balancing classes with under sampling...")
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X, y)
    return [X_under, y_under]


def balance_classes_under_over_mixed(X, y):
    print("Balancing classes with over and under sampling...")
    over = RandomOverSampler(sampling_strategy='auto')
    X, y = over.fit_resample(X, y)
    under = RandomUnderSampler(sampling_strategy='majority')
    X, y = under.fit_resample(X, y)
    return [X, y]


def balance_classes_smote(X, y):
    print("Balancing classes with SMOTE...")
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X, y)
    return [X_smote, y_smote]


def perform_corpus_balance(X, y, balance):
    if balance == 'over':
        [X, y] = balance_classes_oversampling(X, y)
    elif balance == 'under':
        [X, y] = balance_classes_undersampling(X, y)
    elif balance == 'mixed':
        [X, y] = balance_classes_under_over_mixed(X, y)
    elif balance == 'smote':
        [X, y] = balance_classes_smote(X, y)

    return [X, y]
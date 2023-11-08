import read_data as rd
import vectorizer as vt
from datetime import datetime, timezone
import init_config as init
import pandas as pd
import balance_corpus as ba
import classifiers.classifiers as cl
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score
import feature_selection as fs
import utils as ut
import numpy as np
import time
import gc
import argparse
from sklearn.metrics import classification_report
import classifiers.svc as svc


# dataset_filename = 'exported_emo_big5.csv_AgglomerativeClustering_3'
dataset_filename = 'exported_emo_big5_norm.csv_KMeans_2_7'


import warnings
warnings.filterwarnings('ignore')


def run(is_test, is_balance, which_models):
    utc_dt = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS starting at {}".format(utc_dt.astimezone().isoformat()))

    param_dict = init.get_general_parameters(is_test)
    result_map = init.get_result_map()

    # read files and create corpus
    data_train = rd.create_corpus('dataset/' + dataset_filename + '.csv')

    # print(data_train)

    # [X, y] = vt.tfidf_vectorizer(data_train, ngram)
    X = pd.DataFrame(data_train[['fear','anger','anticipation','trust','surprise','sadness',
                                 'disgust','joy','o_score','c_score','e_score','a_score','n_score']])
    y = pd.DataFrame(data_train['cluster'])

    models = cl.get_models(which_models)

    gc.collect()

    print("Running models...")
    for item in models.items():
        hyperparams = item[1]
        model = item[0]
        model_name = model.__class__.__name__

        gc.collect()

        for balance in param_dict['balance']:
            if is_balance == 'True':
                (X, y) = ba.perform_corpus_balance(X, y, balance)
            else:
                balance = 'not-balanced'

            gc.collect()

            print("Folding with KFold...")
            for f in param_dict['folds']:
                random_state = np.random.seed(None)
                cv = KFold(n_splits=f, shuffle=True)

                print("Train test split...")
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=random_state)

                gc.collect()

                for train_index, test_index in cv.split(X_train, y_train):
                    X_train_cv = X.iloc[train_index]
                    X_test_cv = X.iloc[test_index]
                    y_train_cv = y.iloc[train_index]
                    y_test_cv = y.iloc[test_index]

                    for fea in param_dict['feature_selection']:
                        if fea != 'none':
                            print("Performing feature selection...")
                            (X_train_cv, X_test_cv) = fs.perform_features_selection(fea, X_train_cv,
                                                                                 X_test_cv,
                                                                                 y_train_cv.astype('int'))
                        else:
                            start = time.time()
                            print("Running model {}".format(model_name))
                            print(X_train_cv.shape)
                            print(y_train_cv.shape)

                            model = model.fit(X_train_cv, y_train_cv.astype('int').values.ravel())

                            gc.collect()

                            # grid_search = ut.perform_grid_search(model, hyperparams, cv, X_train_cv, y_train_cv)
                            end = time.time()
                            # best_model = grid_search.best_estimator_
                            print("Training time: {}".format(end - start))

                            gc.collect()

                            start = time.time()
                            metrics = cross_validate(estimator=model,
                                                     X=X_train_cv.values,
                                                     y=y_train_cv.astype('int').values.ravel(),
                                                     cv=cv,
                                                     scoring=init.get_default_scoring(),
                                                     error_score="raise")

                            # print(metrics)

                            end = time.time()
                            print("Cross validate time: {}".format(end - start))

                            # model_svm_acc = cross_val_score(estimator=best_model, X=X_train_cv,
                            #                                 y=y_train_cv.astype('int').values.ravel(),
                            #                                 cv=cv, n_jobs=-1)
                            # print(np.mean(model_svm_acc))

                            result_map["Algorithm"].append(model_name)
                            result_map["Accuracy"].append(round(np.mean(metrics['test_accuracy']), 4))
                            result_map["Precision"].append(round(np.mean(metrics['test_precision']), 4))
                            result_map["Recall"].append(round(np.mean(metrics['test_recall']), 4))
                            result_map["F1 Score"].append(round(np.mean(metrics['test_f1_score']), 4))
                            result_map["AUC"].append(round(np.mean(metrics['test_roc_auc']), 4))
                            # result_map["Ngram"].append(ngram)
                            # result_map["Vect. Strategy"].append('TF-IDF')
                            result_map["Bal. Strategy"].append(balance)
                            # result_map["% of Features"].append(p)
                            result_map["Folds"].append(f)
                            result_map["Feat. Selec. Strategy"].append(fea)
                            # result_map["Hyper Params."].append(grid_search.best_params_)
                            result_map["Model"].append(model)
                            pd.set_option('display.max_colwidth', None)
                            pd.set_option('display.max_columns', None)
                            temp_df = pd.DataFrame(result_map)
                            print(temp_df.tail(1).T)
                            #
                            # print(X_test_cv.shape)
                            # print(y_test_cv.shape)
                            # print(y_test_cv.value_counts())

                            # predictions = best_model.predict(X_test_cv)
                            # print(classification_report(y_test_cv.astype('int'), predictions))

                            gc.collect()

                            predictions = model.predict(X_test_cv)
                            print(classification_report(y_test_cv.astype('int'), predictions))

                            # print(y_test_cv.value_counts())

                            gc.collect()

    result_df = pd.DataFrame(result_map)
    result_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    ut.save_df_to_csv(result_df, dataset_filename)
    ut.save_best_model(result_df, dataset_filename)

    utc_dtf = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS ending at {}".format(utc_dtf.astimezone().isoformat()))

    utc_diff = utc_dtf - utc_dt
    minutes = divmod(utc_diff.seconds, 60)
    print('Time spent: ', minutes[0], 'minutes', minutes[1], 'seconds')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--is_test", help="Is the script running in TEST mode?")
    parser.add_argument("-m", "--models", help="Which models should run?",
                        choices=['svc', 'logistic-regression', 'random-forest',
                                 'multinomial-nb', 'bernoulli-nb', 'gaussian-nb', 'complement-nb',
                                 'decision-tree', 'mlp-classifier'])
    parser.add_argument("-b", "--is_balance",
                        help="Do you wish the script to perform balance strategies (SMOTE, UnderSampling, etc) for the dataset?")

    args = parser.parse_args()
    config = vars(args)
    is_test = config['is_test']
    models = config['models']
    is_balance = config['is_balance']

    run(is_test=is_test, which_models=[models], is_balance=is_balance)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

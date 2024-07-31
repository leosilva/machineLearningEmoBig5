import read_data as rd
import vectorizer as vt
from datetime import datetime, timezone
import init_config as init
import sklearn.metrics as me
import pandas as pd
import balance_corpus as ba
import classifiers.classifiers as cl
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score, StratifiedKFold
import feature_selection as fs
import utils as ut
import numpy as np
import time
import gc
import argparse
from sklearn.metrics import classification_report
import classifiers.svc as svc

import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.ERROR)

# -----------------------------------------
# ESCOLHER AQUI PARA QUAL BASE VAI EXECUTAR
# -----------------------------------------
datasets = [
    # 'exported_emo_big5.csv_AgglomerativeClustering_3',
    'exported_emo_big5_norm.csv_KMeans_2_7',
    # 'exported_emo_big5.csv_KMeans_2_7'
]

# -------------------------------------------------------------
# ESCOLHER AQUI QUAL A PORCENTAGEM A SER CONSIDERADA DO DATASET
# -------------------------------------------------------------
perc = [
    # 10,
    # 15,
    # 20,
    # 25,
    # 30,
    # 35,
    # 40,
    # 45,
    # 50,
    # 55,
    # 60,
    # 65,
    # 70,
    # 75,
    # 80,
    # 85,
    # 90,
    # 95
    100
]


import warnings
warnings.filterwarnings('ignore')


def run(is_test, is_balance, which_models):
    utc_dt = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS starting at {}".format(utc_dt.astimezone().isoformat()))

    param_dict = init.get_general_parameters(is_test)
    # result_map = init.get_result_map()

    for dataset_filename in datasets:
        result_map = init.get_result_map()
        for e in range(0,5):
            print("Execution ", (e + 1))

            for p in perc:
                # read files and create corpus
                if p == 100:
                    data_train = rd.create_corpus('dataset/' + dataset_filename + '_train.csv')
                    data_val = rd.create_corpus('dataset/' + dataset_filename + '_test.csv')
                else:
                    data_train = rd.create_corpus('dataset/' + dataset_filename + '_' + str(p) + 'perc_train.csv')
                    data_val = rd.create_corpus('dataset/' + dataset_filename + '_' + str(p) + 'perc_test.csv')

                # [X, y] = vt.tfidf_vectorizer(data_train, ngram)
                X = pd.DataFrame(data_train[['fear','anger','anticipation','trust','surprise','sadness',
                                             'disgust','joy','o_score','c_score','e_score','a_score','n_score']])
                y = pd.DataFrame(data_train['cluster'])

                X_val = pd.DataFrame(data_val[['fear','anger','anticipation','trust','surprise','sadness',
                                             'disgust','joy','o_score','c_score','e_score','a_score','n_score']])
                y_val = pd.DataFrame(data_val['cluster'])

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

                        for i in range(0,10):
                            print("train_test_split ", (i+1))
                            for f in param_dict['folds']:
                                print("Folding with KFold...")
                                # random_state = np.random.seed(None)
                                random_state = np.random.randint(1, 1000)
                                # cv = KFold(n_splits=f, shuffle=True)

                                cv = StratifiedKFold(n_splits=f, shuffle=True, random_state=random_state)

                                print("Train test split...")
                                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                                    test_size=0.4,
                                                                                    random_state=random_state)

                                gc.collect()

                                # for train_index, test_index in cv.split(X_train, y_train):
                                #     X_train_cv = X_train.iloc[train_index]
                                #     X_test_cv = X_train.iloc[test_index]
                                #     y_train_cv = y_train.iloc[train_index]
                                #     y_test_cv = y_train.iloc[test_index]

                                for fea in param_dict['feature_selection']:
                                    # fea_per = 100
                                    if fea != 'none':
                                        print("Performing feature selection...")
                                        for fea_per in param_dict['percentage_features']:
                                            num_features = int((len(X.columns) * fea_per) / 100)

                                            for fea_to_include in param_dict["feature_to_include"]:
                                                (X_train_cv_selected, X_test_cv_selected) = fs.perform_features_selection(fea, num_features, X_train,
                                                                                                     X_test,
                                                                                                     y_train.astype('int'),
                                                                                                    fea_to_include)

                                                X_val_cv = X_val[X_train_cv_selected.columns]
                                                X_test_cv = X_test[X_test_cv_selected.columns]

                                                run_model(X_test_cv, X_train_cv_selected, X_val_cv, balance, cv, f, fea, fea_per, model, model_name,
                                                          result_map, y_test, y_train, y_val, random_state, i, e, hyperparams, p)

                                                gc.collect()

                                    else:
                                        run_model(X_test, X_train, X_val, balance, cv, f, fea, fea_per, model, model_name,
                                                  result_map, y_test, y_train, y_val, random_state, i, e, hyperparams, p)

                                        gc.collect()

            result_df = pd.DataFrame(result_map)
            result_df.sort_values(by='Val. Accuracy', ascending=False, inplace=True)

            ut.save_df_to_csv(result_df, dataset_filename, p)
            ut.save_best_model(result_df, dataset_filename, p)

    utc_dtf = datetime.now(timezone.utc)
    print("GENERAL ANALYSIS ending at {}".format(utc_dtf.astimezone().isoformat()))

    utc_diff = utc_dtf - utc_dt
    minutes = divmod(utc_diff.seconds, 60)
    print('Time spent: ', minutes[0], 'minutes', minutes[1], 'seconds')


def run_model(X_test_cv, X_train_cv, X_val, balance, cv, f, fea, fea_per, model, model_name, result_map, y_test_cv, y_train_cv,
              y_val, random_state, iter, e, hyperparams, perc_dataset):
    start = time.time()
    print("Running model ", model_name)
    # print(X_train_cv.shape)
    # print(y_train_cv.shape)

    model = model.fit(X_train_cv, y_train_cv.astype('int').values.ravel())

    gc.collect()
    # grid_search = ut.perform_grid_search(model, hyperparams, cv, X_train_cv, y_train_cv)
    end = time.time()
    # model = grid_search.best_estimator_
    print("Training time: {}".format(end - start))
    gc.collect()

    start = time.time()
    # print(X_train_cv)
    # print(y_train_cv['cluster'].value_counts())

    metrics = cross_validate(estimator=model,
                             X=X_train_cv,
                             y=y_train_cv.astype('int').values.ravel(),
                             cv=cv,
                             scoring=init.get_default_scoring(),
                             error_score="raise")

    # print(metrics)
    end = time.time()
    print("Cross validate time: {}".format(end - start))
    test_predictions = model.predict(X_test_cv)
    class_report_dict_test = classification_report(y_test_cv.astype('int'), test_predictions,
                                                   output_dict=True)

    # print(class_report_dict_test)

    val_predictions = model.predict(X_val)
    class_report_dict_val = classification_report(y_val.astype('int'), val_predictions,
                                                  output_dict=True)
    # model_svm_acc = cross_val_score(estimator=best_model, X=X_train_cv,
    #                                 y=y_train_cv.astype('int').values.ravel(),
    #                                 cv=cv, n_jobs=-1)
    # print(np.mean(model_svm_acc))

    feature_importance = fs.permutation_feature_importance(model, X_train_cv, y_train_cv)

    result_map["Algorithm"].append(model_name)
    result_map["Train Accuracy"].append(round(np.mean(metrics['test_accuracy']), 4))
    result_map["Train Precision"].append(round(np.mean(metrics['test_precision']), 4))
    result_map["Train Recall"].append(round(np.mean(metrics['test_recall']), 4))
    result_map["Train F1 Score"].append(round(np.mean(metrics['test_f1_score']), 4))
    result_map["Train MSE"].append(round(me.mean_squared_error(y_test_cv, test_predictions), 4))

    result_map["Test Accuracy"].append(round(np.mean(class_report_dict_test['accuracy']), 4))
    result_map["Test Precision"].append(
        round(np.mean(class_report_dict_test['weighted avg']['precision']), 4))
    result_map["Test Recall"].append(
        round(np.mean(class_report_dict_test['weighted avg']['recall']), 4))
    result_map["Test F1 Score"].append(
        round(np.mean(class_report_dict_test['weighted avg']['f1-score']), 4))
    result_map["Val. Accuracy"].append(round(np.mean(class_report_dict_val['accuracy']), 4))
    result_map["Val. Precision"].append(
        round(np.mean(class_report_dict_val['weighted avg']['precision']), 4))
    result_map["Val. Recall"].append(
        round(np.mean(class_report_dict_val['weighted avg']['recall']), 4))
    result_map["Val. F1 Score"].append(
        round(np.mean(class_report_dict_val['weighted avg']['f1-score']), 4))
    result_map["Val. MSE"].append(
        round(me.mean_squared_error(y_val, val_predictions), 4))
    result_map["AUC"].append(round(np.mean(metrics['test_roc_auc']), 4))
    # result_map["Ngram"].append(ngram)
    # result_map["Vect. Strategy"].append('TF-IDF')
    result_map["Bal. Strategy"].append(balance)
    result_map["% of Features"].append(fea_per)
    result_map["% of Dataset"].append(perc_dataset)
    result_map["Folds"].append(f)
    result_map["Fold"].append((iter + 1))
    result_map["Execution"].append((e + 1))
    result_map["Feat. Selec. Strategy"].append(fea)
    result_map["Features"].append(list(X_val.columns))
    result_map["Features Importance"].append(feature_importance)
    # result_map["Hyper Params."].append(grid_search.best_params_)
    result_map["Model"].append(model)
    result_map["Random State"].append(random_state)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    temp_df = pd.DataFrame(result_map)
    print(temp_df.tail(1).T)

    # print(X_test_cv.shape)
    # print(y_test_cv.shape)
    # print(y_test_cv.value_counts())
    # predictions = best_model.predict(X_test_cv)
    # print(classification_report(y_test_cv.astype('int'), predictions))
    # gc.collect()
    # predictions = model.predict(X_test_cv)
    # print(classification_report(y_test_cv.astype('int'), predictions))
    # print(y_test_cv.value_counts())

    gc.collect()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--is_test", help="Is the script running in TEST mode?")
    parser.add_argument("-m", "--models", help="Which models should run?")
                        # choices=['knn', 'svc', 'logistic-regression', 'random-forest',
                        #          'multinomial-nb', 'bernoulli-nb', 'gaussian-nb', 'complement-nb',
                        #          'decision-tree', 'mlp-classifier'])
    parser.add_argument("-b", "--is_balance",
                        help="Do you wish the script to perform balance strategies (SMOTE, UnderSampling, etc) for the dataset?")

    parser.add_argument("-s", "--specific_alg",
                        help="Run?")

    args = parser.parse_args()
    config = vars(args)
    is_test = config['is_test']
    models = config['models']
    is_balance = config['is_balance']

    if "," in models:
        models = models.split(',')
    else:
        models = [models]

    run(is_test=is_test, which_models=models, is_balance=is_balance)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

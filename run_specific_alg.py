import classifiers.classifiers as cl
import time
import gc
import numpy as np
import pandas as pd
import utils as ut
import read_data as rd
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score, StratifiedKFold
import init_config as init
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.ERROR)
np.set_printoptions(precision=4)



def run_model(model_name, dataset):
    start = time.time()
    features = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy', 'a_score']

    print("Running model ", model_name)

    if dataset == 'ac':
        data_train = rd.create_corpus('dataset/exported_emo_big5.csv_AgglomerativeClustering_3_50perc_train.csv')
        data_val = rd.create_corpus('dataset/exported_emo_big5.csv_AgglomerativeClustering_3_50perc_test.csv')
    elif dataset == 'km':
        data_train = rd.create_corpus('dataset/exported_emo_big5.csv_KMeans_2_7_50perc_train.csv')
        data_val = rd.create_corpus('dataset/exported_emo_big5.csv_KMeans_2_7_50perc_test.csv')

    X = pd.DataFrame(data_train[features])
    y = pd.DataFrame(data_train['cluster'])

    X_val = pd.DataFrame(data_val[features])
    y_val = pd.DataFrame(data_val['cluster'])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=635)

    print("Train test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=635)

    # print(X_train_cv.shape)
    # print(y_train_cv.shape)
    model_dict = cl.get_models([model_name])
    for item in model_dict.items():
        hyperparams = item[1]
        model = item[0]

    model = model.fit(X_train, y_train.astype('int').values.ravel())
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
                             X=X_train,
                             y=y_train.astype('int').values.ravel(),
                             cv=cv,
                             scoring=init.get_default_scoring(),
                             error_score="raise")

    # print(metrics)
    end = time.time()
    print("Cross validate time: {}".format(end - start))
    test_predictions = model.predict(X_test)
    class_report_dict_test = classification_report(y_test.astype('int'), test_predictions, digits=4)

    # sns.heatmap(pd.DataFrame(class_report_dict_test).iloc[:-1, :].T, annot=True, fmt=".4g").set(title='Classification Report - Test Dataset')
    # plt.show()

    print("Test classification report")
    print(class_report_dict_test)

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        # display_labels=class_names,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title("Confusion Matrix - Test Dataset")

    print("Confusion Matrix - Test dataset")
    print(disp.confusion_matrix)

    # plt.show()


    val_predictions = model.predict(X_val)
    class_report_dict_val = classification_report(y_val.astype('int'), val_predictions, digits=4)

    # sns.heatmap(pd.DataFrame(class_report_dict_val).iloc[:-1, :].T, annot=True, fmt=".4g").set(
    #     title='Classification Report - Validation Dataset')
    # plt.show()

    print("Validation classification report")
    print(class_report_dict_val)

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_val,
        y_val,
        # display_labels=class_names,
        cmap=plt.cm.Blues,
    )
    disp.ax_.set_title("Confusion Matrix - Validation Dataset")

    print("Confusion Matrix - Validation dataset")
    print(disp.confusion_matrix)

    # plt.show()

    gc.collect()


run_model('random-forest', 'ac')
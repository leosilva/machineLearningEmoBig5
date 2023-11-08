import pandas as pd
import preprocessing as pp
import numpy as np

def read_data(file_train):
    data_train = pd.read_csv(file_train, sep=',')
    return data_train


def create_corpus(file_train):
    print("Creating corpus...")
    data_train = read_data(file_train)
    # data_train.columns = ['coders_classification', 'text', 'date', 'id', 'id_user', 'participant']

    return data_train
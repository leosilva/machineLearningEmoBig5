import pandas as pd
import preprocessing as pp
import numpy as np

def read_data(file_train, file_test):
    data_train = pd.read_csv(file_train, sep=';')
    data_test = pd.read_csv(file_test, sep=';')
    return [data_train, data_test]


def create_corpus(file_train, file_test):
    print("Creating corpus...")
    (data_train, data_test) = read_data(file_train, file_test)
    data_train.columns = ['coders_classification', 'text', 'date', 'id', 'id_user', 'participant']
    data_test.columns = ['coders_classification', 'text', 'date', 'id', 'id_user', 'participant']
    temp_data_train = pp.create_emoji_features(data_train)
    temp_data_test = pp.create_emoji_features(data_test)

    data_train_test = [temp_data_train, temp_data_test]

    result = []

    for i in data_train_test:
        temp_data = i
        # normalize corpus
        temp_data['text_clean'] = pp.normalize_corpus(corpus=temp_data['text'],
                                                   html_stripping=True,
                                                   accented_char_removal=True,
                                                   text_lower_case=True,
                                                   text_lemmatization=True,
                                                   text_stemming=True,
                                                   special_char_removal=True,
                                                   remove_digits=True,
                                                   remove_repeated_char=False,
                                                   remove_urls=True,
                                                   remove_tt_handles=True,
                                                   remove_tt_hashtag=True,
                                                   stopword_removal=True)

        for t in temp_data.index:
            tweet = temp_data.loc[t]
            temp_data.at[t, 'text_clean'] = tweet.text_clean + ' ' + tweet.emojis

        conditions = [
            (temp_data['coders_classification'] == 'Negativo'),
            (temp_data['coders_classification'] == 'Neutro'),
            (temp_data['coders_classification'] == 'Positivo')
        ]

        values = [-1, 0, 1]

        temp_data['coders_classification'] = np.select(conditions, values)
        result.append(temp_data)

    return result
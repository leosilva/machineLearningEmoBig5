import read_data as rd
import pandas as pd


# dataset_filename = 'exported_emo_big5.csv_AgglomerativeClustering_3'
# dataset_filename = 'exported_emo_big5_norm.csv_KMeans_2_7'
# dataset_filename = 'exported_emo_big5.csv_KMeans_2_7'

datasets = [
    'exported_emo_big5.csv_AgglomerativeClustering_3',
    # 'exported_emo_big5_norm.csv_KMeans_2_7',
    # 'exported_emo_big5.csv_KMeans_2_7'
]

perc = [
    # 20,
    # 25,
    # 30,
    # 35,
    # 40
    # 45,
    # 50,
    # 55,
    # 60,
    # 65,
    # 70,
    75,
    80,
    85,
    90,
    95
]

for dataset_filename in datasets:
    for p in perc:
        data_train = rd.create_corpus('dataset/' + dataset_filename + '.csv')
        print("total dataset: " + str(len(data_train)))

        zero_sample = data_train.query("cluster == 0").sample(frac=(p/100))
        one_sample = data_train.query("cluster == 1").sample(frac=(p/100))
        two_sample = data_train.query("cluster == 2").sample(frac=(p/100))

        print("zero sample: " + str(len(zero_sample)))
        print("one sample: " + str(len(one_sample)))
        print("two sample: " + str(len(two_sample)))

        test_result_df = zero_sample.append(one_sample).append(two_sample)

        print("ten percent dataset: ", len(test_result_df))
        print(test_result_df['cluster'].value_counts())

        zero_sample = test_result_df.query("cluster == 0").sample(frac=0.3)
        one_sample = test_result_df.query("cluster == 1").sample(frac=0.3)
        two_sample = test_result_df.query("cluster == 2").sample(frac=0.3)

        validation_result_df = zero_sample.append(one_sample).append(two_sample)

        print("validation dataset: ", len(validation_result_df))
        print(validation_result_df['cluster'].value_counts())

        # print(test_result_df.head())

        idx_final_test_df = validation_result_df.index.tolist()
        data_train_filtered = test_result_df.drop(index=idx_final_test_df)

        X_train = pd.DataFrame(data_train_filtered.iloc[:,:-1])
        y_train = pd.DataFrame(data_train_filtered['cluster'])
        X_test = pd.DataFrame(validation_result_df.iloc[:,:-1])
        y_test = pd.DataFrame(validation_result_df['cluster'])

        # print(X_train.head())
        # print(y_train.head())

        X_train_complete = X_train.join(y_train)
        X_test_complete = X_test.join(y_test)

        print(X_train_complete['cluster'].value_counts())
        print(X_test_complete['cluster'].value_counts())

        X_train_complete.to_csv('dataset/' + dataset_filename + '_' + str(p) + 'perc_train.csv', sep=',')
        X_test_complete.to_csv('dataset/' + dataset_filename + '_' + str(p) + 'perc_test.csv', sep=',')
from sklearn.model_selection import GridSearchCV
import joblib
import os


execution = "percent_execution"


def perform_grid_search(model, params, cv, X_train_selected, y_train_cv):
    grid_search = GridSearchCV(model, param_grid=params, cv=cv, refit = True, n_jobs=-1)
    grid_search.fit(X_train_selected, y_train_cv.astype('int').values.ravel())

    return grid_search


def save_df_to_csv(df, folder, p):
    best_model = df.head(1)
    model_name = best_model['Algorithm'].values[0]

    if not os.path.exists('best_models/' + execution + '/' + str(p) + '/'):
        os.makedirs('best_models/' + execution + '/' + str(p) + '/')

    filename = 'best_models/' + execution + '/' + str(p) + '/' + folder + '_' + model_name + '.csv'
    print(filename)
    df.to_csv(filename, index=None, sep=',', mode='w')


def save_best_model(result_df, folder):
    best_model = result_df.head(1)
    m = best_model['Model'].iloc[0]
    model_name = best_model['Algorithm'].values[0]

    if not os.path.exists('best_models/' + execution + '/' + folder + '/'):
        os.makedirs('best_models/' + execution + '/' + folder + '/')

    filename = 'best_models/' + execution + '/' + folder + '/' + 'best_model_' + model_name + '.pkl'
    joblib.dump(m, filename)
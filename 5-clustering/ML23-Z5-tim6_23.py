import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

NUM_COL_NAMES = ['BDP', 'Izvoz']
CAT_COL_NAMES = ['Religija', 'More']

def load_file_paths_from_cmd() -> (str, str):
    if len(sys.argv) > 2:
        train_directory_name = sys.argv[1]
        test_directory_name = sys.argv[2]
        return train_directory_name, test_directory_name
    return "train.csv", "test_preview.csv"

def read_data_form_file(filepath: str):
    return pd.read_csv(filepath)

def preprocessing_dfs(train_df):
    num_col_imputer = SimpleImputer(strategy='mean')
    train_df[NUM_COL_NAMES] = num_col_imputer.fit_transform(train_df[NUM_COL_NAMES])

    cat_col_imputer = SimpleImputer(strategy='most_frequent')
    train_df[CAT_COL_NAMES] = cat_col_imputer.fit_transform(train_df[CAT_COL_NAMES])

    train_df.dropna(inplace=True)
    return train_df

def remove_outliers(train_df):
    train_df.dropna(thresh=4, inplace=True)
    conditions = (train_df['BDP'] >= 100000) | (train_df['Izvoz'] >= 120)
    return train_df[~conditions]

def get_X_and_Y_from_df(df):
    return df.drop(['Region', "Inflacija"], axis=1), df['Region']

def transform_Xs(train_X, test_X):
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(), CAT_COL_NAMES),
        ('num', StandardScaler(), NUM_COL_NAMES)
    ])
    return transformer.fit_transform(train_X), transformer.transform(test_X)


def transform_Ys(train_Y, test_Y):
    transformer = LabelEncoder()
    return transformer.fit_transform(train_Y), transformer.transform(test_Y)



def calculate_v_metric(labels_true, labels_pred):

    return metrics.v_measure_score(labels_true, labels_pred)

def to_polynomial(x, degree):
    for deg in range(2, degree + 1):
        x = np.hstack((x, x ** deg))
    return x

def main():
    train_file_name, test_file_name = load_file_paths_from_cmd()

    train_df = read_data_form_file(train_file_name)
    test_df = read_data_form_file(test_file_name)

    train_df = remove_outliers(train_df)

    train_df = preprocessing_dfs(train_df)

    train_X, train_Y = get_X_and_Y_from_df(train_df)
    test_X, test_Y = get_X_and_Y_from_df(test_df)

    train_X, test_X = transform_Xs(train_X, test_X)
    train_Y, test_Y = train_Y.values, test_Y.values

    train_X = to_polynomial(train_X, 2)
    test_X = to_polynomial(test_X, 2)

    gmm = GaussianMixture(n_components=4, random_state=0,n_init=5, covariance_type='tied')
    gmm.fit(train_X)

    labels_predict = gmm.predict(test_X)
    print(calculate_v_metric(test_Y, labels_predict))


if __name__ == "__main__":
    main()

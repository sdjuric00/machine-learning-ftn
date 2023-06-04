import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

NUM_COL_NAMES = ['masa', 'godina', 'starost']
CAT_COL_NAMES = ['jastuk', 'pojas', 'ceoni', 'tip']


def load_file_paths_from_cmd() -> (str, str):
    if len(sys.argv) > 2:
        train_directory_name = sys.argv[1]
        test_directory_name = sys.argv[2]
        return train_directory_name, test_directory_name
    return "train.csv", "test_preview.csv"


def read_data_form_file(filepath: str):
    return pd.read_csv(filepath)


def remove_outliers(train_df):
    train_df.dropna(thresh=7, inplace=True)
    return train_df


def preprocessing_dfs(train_df, test_df):
    num_col_imputer = SimpleImputer(strategy='median')
    train_df[NUM_COL_NAMES] = num_col_imputer.fit_transform(train_df[NUM_COL_NAMES])
    test_df[NUM_COL_NAMES] = num_col_imputer.transform(test_df[NUM_COL_NAMES])

    cat_col_imputer = SimpleImputer(strategy='most_frequent')
    train_df[CAT_COL_NAMES] = cat_col_imputer.fit_transform(train_df[CAT_COL_NAMES])
    test_df[CAT_COL_NAMES] = cat_col_imputer.transform(test_df[CAT_COL_NAMES])

    train_df.dropna(inplace=True)
    return train_df, test_df


def get_X_and_Y_from_df(df):
    return df.drop(['pol', 'ishod'], axis=1), df['ishod']


def transform_Xs(train_X, test_X):
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(), CAT_COL_NAMES),
        ('num', StandardScaler(), NUM_COL_NAMES)
    ])
    return transformer.fit_transform(train_X), transformer.transform(test_X)


def fit(X, Y):
    ada_clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=42, class_weight='balanced'),
        learning_rate=0.5,
        n_estimators=150,
        random_state=42,
        algorithm="SAMME"
    )

    ada_clf.fit(X, Y)
    return ada_clf


def predict(model, X, Y):
    print(f1_score(Y, model.predict(X), average='macro'))


def apply_PCA(train_X, test_X, n_components=None):
    transformer = PCA(n_components=n_components)
    transformer.fit(train_X)

    return transformer.transform(train_X), transformer.transform(test_X)

def main():
    train_fp, test_fp = load_file_paths_from_cmd()
    train_df = read_data_form_file(train_fp)
    test_df = read_data_form_file(test_fp)

    train_df = remove_outliers(train_df)

    train_df, test_df = preprocessing_dfs(train_df, test_df)

    train_X, train_Y = get_X_and_Y_from_df(train_df)
    test_X, test_Y = get_X_and_Y_from_df(test_df)

    train_X, test_X = transform_Xs(train_X, test_X)
    train_X, test_X = apply_PCA(train_X, test_X, n_components=4)
    train_Y, test_Y = train_Y.values, test_Y.values

    model = fit(train_X, train_Y)
    predict(model, test_X, test_Y)


if __name__ == "__main__":
    main()

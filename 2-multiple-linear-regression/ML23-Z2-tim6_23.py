from math import sqrt
import sys
from typing import List
import numpy as np
import pandas as pd


def load_file_paths_from_cmd() -> (str, str):
    if len(sys.argv) > 2:
        train_directory_name = sys.argv[1]
        test_directory_name = sys.argv[2]
        return train_directory_name, test_directory_name
    return "train.csv", "test_preview.csv"


def read_file(filepath: str):
    return pd.read_csv(filepath)


def remove_unused_columns(data_frame, col_names: List[str]):
    return data_frame.drop(col_names, axis=1)


def one_hot_encode(df_train, df_test, col_names: List[str]):
    for cl_name in col_names:
        labels = df_train[cl_name].unique()
        labels_encode_schemas = {lab1: {lab2: 1 if lab2 == lab1 else 0 for lab2 in labels} for lab1 in labels}
        for lab in labels:
            df_train[cl_name + lab] = df_train[cl_name].map(labels_encode_schemas[lab])
            df_test[cl_name + lab] = df_test[cl_name].map(labels_encode_schemas[lab])
        df_train = df_train.drop([cl_name], axis=1)
        df_test = df_test.drop([cl_name], axis=1)

    return df_train, df_test


def standardize(df_train, df_test, col_names: List[str]):
    for cl_name in col_names:
        mean = df_train[cl_name].mean()
        standard_deviation = df_train[cl_name].std()
        df_train[cl_name] = (df_train[cl_name] - mean) / standard_deviation
        df_test[cl_name] = (df_test[cl_name] - mean) / standard_deviation
    return df_train, df_test


def get_x_and_y_from_dataframe(dataframe, y_col_name: str):
    return dataframe.drop([y_col_name], axis=1), dataframe[y_col_name]


def to_polynomial(x, degree):
    for deg in range(2, degree + 1):
        x = np.hstack((x, x ** deg))
    return x


def fit(x_train, y_train, LR=0.21, ALPHA=0.3, EPOCHS=9000):
    # Lasso
    rows, columns = x_train.shape
    coefficients = np.zeros(columns)
    for _ in range(EPOCHS):
        y_pred = np.dot(x_train, coefficients)
        error = y_pred - y_train
        gradients = (1 / rows) * np.dot(x_train.T, error) + ALPHA * np.sign(coefficients)
        coefficients -= LR * gradients
    return coefficients


def predict(x_test, coefficients):
    return np.dot(x_test, coefficients)


def print_rmse(Y_pred, Y_actual) -> None:
    print(sqrt(sum((Y_pred - Y_actual) ** 2) / len(Y_actual)))


def main():
    UNUSED_COLUMNS = ["transmission", "color"]
    CATEGORICAL_COLUMNS = ['make', 'category', 'fuel']
    NUMERICAL_COLUMNS = ['year', "engine_size", 'mileage']
    PRICE_COLUMN_NAME = "price"

    train_fp, test_fp = load_file_paths_from_cmd()

    df_train = remove_unused_columns(read_file(train_fp), UNUSED_COLUMNS)
    df_test = remove_unused_columns(read_file(test_fp), UNUSED_COLUMNS)

    df_train, df_test = one_hot_encode(df_train, df_test, CATEGORICAL_COLUMNS)
    df_train, df_test = standardize(df_train, df_test, NUMERICAL_COLUMNS)

    x_train, y_train = get_x_and_y_from_dataframe(df_train, PRICE_COLUMN_NAME)
    x_test, y_test = get_x_and_y_from_dataframe(df_test, PRICE_COLUMN_NAME)

    x_train = to_polynomial(x_train, 2)
    x_test = to_polynomial(x_test, 2)

    coefficients = fit(x_train, y_train)
    y_prediction = predict(x_test, coefficients)

    print_rmse(y_prediction, y_test)


if __name__ == "__main__":
    main()

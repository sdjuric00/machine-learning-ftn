import sys
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_file_paths_from_cmd() -> (str, str):
    """
    :return: first -> file path to training data, second -> file path to test data

    If at least 2 system arguments are passed, they are returned assuming to be file
        paths to data, if less than 2 are given, defaults are returned
    """
    if len(sys.argv) > 2:
        train_directory_name = sys.argv[1]
        test_directory_name = sys.argv[2]
        return train_directory_name, test_directory_name
    return "train.csv", "test_preview.csv"


def read_xy_from_file(file_path: str):
    """
    :return: (x_col, y_col)
    """
    data = pd.read_csv(file_path)
    return data.iloc[:, 1], data.iloc[:, 0]


def transform_outliers(X, Y) -> None:
    """
    :param X: values of 'x' column
    :param Y: values of 'y' column
    """
    mask = (Y > 850) | ((X > 5) & (Y <= 160))
    X.loc[mask] = np.log(X.loc[mask])
    Y.loc[mask] = np.log(Y.loc[mask])


def fit(X, Y) -> (float, float, float, float, float, float):
    """
    :param X: values of 'x' column
    :param Y: values of 'y' column
    :return: coefficients of f(x)

    LR -> learning rate
    LR_BY_N -> learning rate divided by number of values in an array (used in theta value adjusting formula)
    X_2nd, X_3rd, X_4th, X_5th -> used in formula for theta values adjusting so made as constants
    """
    EPOCHS = 9000
    LR = 0.000000031
    LR_BY_N = LR / len(X)
    theta0, theta1, theta2, theta3, theta4, theta5, theta6 = 0, 0, 0, 0, 0, 0, 0
    X_2nd, X_3rd, X_4th, X_5th = X ** 2, X ** 3, X ** 4, X ** 5

    for _ in range(EPOCHS):
        Y_pred = theta5 * X_5th + theta4 * X_4th + theta3 * X_3rd + theta2 * X_2nd + theta1 * X + theta0
        Y_dif = Y_pred - Y

        theta0 -= LR_BY_N * sum(Y_dif)
        theta1 -= LR_BY_N * sum(Y_dif * X)
        theta2 -= LR_BY_N * sum(Y_dif * X_2nd)
        theta3 -= LR_BY_N * sum(Y_dif * X_3rd)
        theta4 -= LR_BY_N * sum(Y_dif * X_4th)
        theta5 -= LR_BY_N * sum(Y_dif * X_5th)

    return theta0, theta1, theta2, theta3, theta4, theta5


def predict(X, theta: (float, float, float)):
    """
    :param X: 'x' column values by which the 'y' values are predicted
    :param theta: coefficients used in predicting formula
    :return: 'y' value calculated by the formula 'y = sum_0-i(theta[i] + X ** i)'
    """
    return theta[5] * X ** 5 + theta[4] * X ** 4 + theta[3] * X ** 3 + theta[2] * X ** 2 + theta[1] * X + theta[0]


def print_rmse(Y_pred, Y_actual) -> None:
    """
    :param Y_pred: array of PREDICTED 'y' column values
    :param Y_actual: array of ACTUAL 'y' column values
    """
    print(sqrt(sum((Y_pred - Y_actual) ** 2) / len(Y_actual)))


def plotting(X_fit, Y_fit, X_test, Y_test, theta) -> None:
    plt.scatter(X_fit, Y_fit, c="blue")
    plt.scatter(X_test, Y_test, c="orange")

    x_seq = np.linspace(X_fit.min(), X_fit.max(), 100)
    y_seq = theta[5] * x_seq ** 5 + theta[4] * x_seq ** 4 + theta[3] * x_seq ** 3 + theta[2] * x_seq ** 2 + theta[1] * x_seq + theta[0]

    # plt.plot(x_seq, model(x_seq))
    plt.plot(x_seq, y_seq)
    plt.show()


def main() -> None:
    """
    1. File path are loaded
    2. 'x' and 'y' values are read from files
    3. Outliers are removed from the dataset
    4. Theta coefficients are calculated using training data
    5. Using theta coefficients and 'x' column of test data 'y' values are
        predicted and 'rmse' is calculated (and printed) by comparing actual
        'y' values in from test data and predicted ones
    """
    train_data_file_path, test_data_file_path = load_file_paths_from_cmd()

    X_fit, Y_fit = read_xy_from_file(train_data_file_path)
    X_test, Y_test = read_xy_from_file(test_data_file_path)

    X_train, X_test, Y_train, Y_test = train_test_split(X_fit, Y_fit, test_size=0.2, shuffle=True)

    transform_outliers(X_train, Y_train)
    transform_outliers(X_test, Y_test)

    theta = fit(X_train, Y_train)
    print_rmse(predict(X_test, theta), Y_test)

    plotting(X_train, Y_train, X_test, Y_test, theta)

if __name__ == "__main__":
    main()

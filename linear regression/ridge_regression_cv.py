import numpy as np
import pandas as pd

DATA_PATH = r"./data/"

data_train = {
    "X": np.genfromtxt(DATA_PATH + "data_train_X.csv", delimiter=","),
    "t": np.genfromtxt(DATA_PATH + "data_train_y.csv", delimiter=",")
}
data_test = {
    "X": np.genfromtxt(DATA_PATH + "data_test_X.csv", delimiter=","),
    "t": np.genfromtxt(DATA_PATH + "data_test_y.csv", delimiter=",")
}

df_train = pd.DataFrame(data_train["X"])
df_train["t"] = data_train["t"]

df_test = pd.DataFrame(data_test["X"])
df_test["t"] = data_test["t"]


def shuffle_data(data: pd.DataFrame):
    """Function that returns a randomly permuted copy of data."""
    permutation = np.random.permutation(data.shape[0])
    return data.iloc[permutation]


def split_data(data: pd.DataFrame, num_folds: int, fold: int):
    """Function that returns one fold of validation and num_folds - 1 folds of training data.
    """
    assert fold in range(1, num_folds + 1)
    start = data.shape[0] / num_folds * (fold - 1)
    end = start + data.shape[0] / num_folds
    # mask off the validation.
    mask = data.index.isin(np.arange(start, end))
    return data[mask], data[~mask]


def train_model(data: pd.DataFrame, lambd: float):
    """Function that returns the coefficient vector from a trained ridge regression
    model with the penalty constant lambd."""
    X = data.drop("t", axis=1)
    t = data["t"]
    w_ridge = np.linalg.inv(np.dot(X.T, X) + lambd * X.shape[0] * np.identity(X.shape[1]))
    return w_ridge


def predict(data, model):
    """Function returns predication based on data and model."""
    return np.dot(data.drop("t", axis=1), model)


def loss(data, model):
    """Function calculates the loss based on data and model."""
    return np.power(np.linalg.norm(np.dot(data.drop("t", axis=1), model) - data["t"]), 2) / (2 * data.shape[0])


def cross_validation(data: pd.DataFrame, num_folds: int, lambd_seq: list):
    """Function that computes loss for models with lambda specified in lambd_seq.
    It returns a vector of len(lambd_seq)."""
    cv_error = []
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lamdb = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lamdb)
            cv_loss_land += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


if __name__ == "__main__":
    pass
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
df_train[-1] = data_train["t"]

df_test = pd.DataFrame(data_test["X"])
df_test[-1] = data_test["t"]


def shuffle_data(data: pd.DataFrame):
    """Function that returns a randomly permuted copy of data."""
    permutation = np.random.permutation(data.shape[0])
    return data.iloc[permutation]


def split_data(data, num_folds, fold):
    """Function that returns one fold of validation and num_folds - 1 folds of training data.
    """
    assert fold in range(1, num_folds + 1)
    start = data.shape[0] / num_folds * (fold - 1)
    end = start + data.shape[0] / num_folds
    # mask off the validation.
    mask = data.index.isin(np.arange(start, end))
    return data[mask], data[~mask]


def train_model(data, lambd):
    """Function that returns the coefficient vector from a trained ridge regression
    model with the penalty constant lambd."""
    pass


def predict(data, model):
    """Function returns predication based on data and model."""
    pass


def loss(data, model):
    """Function calculates the loss based on data and model."""
    pass


def cross_validation(data, num_folds, lambd_seq):
    """Function that computes loss for models with lambda specified in lambd_seq.
    It returns a vector of len(lambd_seq)."""
    pass


if __name__ == "__main__":
    pass
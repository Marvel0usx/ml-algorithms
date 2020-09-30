import numpy as np
import pandas as pd

file_path = r"../data/"

data_train = {
    "X": np.genfromtxt(file_path + "data_train_X.csv", delimiter=","),
    "t": np.genfromtxt(file_path + "data_train_y.csv", delimiter=",")
}
data_test = {
    "X": np.genfromtxt(file_path + "data_test_X.csv", delimiter=","),
    "t": np.genfromtxt(file_path + "data_test_y.csv", delimiter=",")
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
    mask = np.arange(start, end)
    return data.loc[data.index.isin(mask)], data.loc[~data.index.isin(mask)]    # data.loc selects rows


def train_model(data: pd.DataFrame, lambd: float):
    """Function that returns the coefficient vector from a trained ridge regression
    model with the penalty constant lambd."""
    X = data.drop("t", axis=1)
    t = data["t"]
    # Direct method to find optimal weight.
    w_ridge = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lambd * X.shape[0] * np.identity(X.shape[1])), X.T), t)
    return w_ridge


def predict(data, model):
    """Function returns predication based on data and model."""
    return np.dot(data.drop("t", axis=1), model)    # data.drop gives X


def loss(data, model):
    """Function calculates the loss based on data and model."""
    return np.sum(np.square(predict(data, model) - data["t"])) / (2 * data.shape[0])


def cross_validation(data: pd.DataFrame, num_folds: int, lambd_seq):
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
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


def lambd_specific_error(lambd_seq):
    """Function that computes error corresponding to each lambd in lambd_seq."""
    error_list = []
    lambd_list = []
    for lambd in lambd_seq:
        model = train_model(df_train, lambd)
        train_error = loss(df_train, model)
        test_error = loss(df_test, model)
        error_list.append(train_error)
        error_list.append(test_error)
        lambd_list.extend([lambd] * 2)
    error_df = pd.DataFrame({
        "lambd": lambd_list,
        "type": ["train", "test"] * len(lambd_seq),
        "error": error_list
    })
    return error_df


def plot_errors(df_errors):
    """Function that plots errors."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.lineplot(data=df_errors, x="lambd", y="error", hue="type", style="type", markers=True, dashes=False)
    plt.show()


if __name__ == "__main__":
    lambd_seq = np.linspace(0.00005, 0.005, 50)
    fold_5_errors = pd.DataFrame({"type": "5-fold",
                                  "lambd": lambd_seq,
                                  "error": cross_validation(df_train, 5, lambd_seq)})
    fold_10_errors = pd.DataFrame({"type": "10-fold",
                                   "lambd": lambd_seq,
                                   "error": cross_validation(df_train, 10, lambd_seq)})
    # Summarize all errors to dataframe.
    df_errors = lambd_specific_error(lambd_seq).append(fold_5_errors, sort=False).append(fold_10_errors, sort=False)
    # Plot
    plot_errors(df_errors)

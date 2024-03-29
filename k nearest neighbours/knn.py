from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import numpy as np
import pandas as pd

FAKE_HEADER = r"../data/clean_fake.txt"
REAL_HEADER = r"../data/clean_real.txt"

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.5
TEST_SPLIT = 0.5

K_START = 1
K_END = 20


def load_data():
    """Function that victories news headers."""
    fake_headers = open(FAKE_HEADER, "r").readlines()
    real_headers = open(REAL_HEADER, "r").readlines()

    fake_header_vec = CountVectorizer()
    real_headers_vec = CountVectorizer()

    fake_header_vec.fit_transform(fake_headers)
    real_headers_vec.fit_transform(real_headers)

    features = sorted(list(set(fake_header_vec.get_feature_names() +\
                real_headers_vec.get_feature_names())))

    new_fake_header_vec = CountVectorizer(vocabulary=features)
    new_real_header_vec = CountVectorizer(vocabulary=features)

    fake_header_X = new_fake_header_vec.fit_transform(fake_headers).toarray()
    real_header_X = new_real_header_vec.fit_transform(real_headers).toarray()

    dataset_t = np.append(
        np.zeros(fake_header_X.shape[0]),
        np.ones(real_header_X.shape[0])
    )

    dataset_X = np.append(fake_header_X, real_header_X, axis=0)
    dataset = pd.DataFrame(np.concatenate((dataset_X, dataset_t.reshape((dataset_t.shape[0], 1))), axis=1), columns=features + ["t"])

    training_X_mask = np.random.rand(dataset.shape[0]) < TRAIN_SPLIT
    training_X = dataset[training_X_mask]
    remained_X = dataset[~training_X_mask]

    test_validation_mask = np.random.rand(remained_X.shape[0]) < TEST_SPLIT
    testing_X = remained_X[test_validation_mask]
    validating_X = remained_X[~test_validation_mask]

    return training_X, testing_X, validating_X


def select_knn_model(training_X, validating_X, metric=None):
    models = {}
    for k in range(K_START, K_END + 1):
        this = KNeighborsClassifier(n_neighbors=k, metric=[metric, "minkowski"][metric is None])
        this.fit(training_X.iloc[:, :-1], training_X.iloc[:, -1])
        models[k] = [this]
        models[k].append(this.score(training_X.iloc[:, :-1], training_X.iloc[:, -1]))
        models[k].append(this.score(validating_X.iloc[:, :-1], validating_X.iloc[:, -1]))

    best_model = sorted(models.items(), key=lambda m: m[1][2])[0][0]
    return best_model, models


def plot_errors(models):
    data = {
        "k": [*models.keys()] * 2,
        "error_type": ["validation error"] * len(models) + ["training error"] * len(models),
        "error": [e[2] for e in models.values()] + [e[1] for e in models.values()]
    }
    plot_df = pd.DataFrame(data)
    plt = sns.lineplot(data=plot_df, x="k", y="error", hue="error_type", markers=True, dashes=True)
    return plt


def report_best_model_performance(model, data):
    """Function that compute loss based on model and data."""
    score = model.score(data.iloc[:, :-1], data.iloc[:, -1])
    print(f"Testing Score: {score}")


if __name__ == "__main__":
    training_X, testing_X, validating_X = load_data()

    best_model, models = select_knn_model(training_X, validating_X)
    best_model_2, models_2 = select_knn_model(training_X, validating_X, "cosine")
    plot_errors(models)
    plot_errors(models_2)

    # Report best models' performance.
    report_best_model_performance(models[best_model][0], testing_X)
    report_best_model_performance(models_2[best_model_2][0], testing_X)

    import pickle
    pickle.dump(models, open("knn-minkowski.dat", "wb"))
    pickle.dump(models_2, open("knn-cosine.dat", "wb"))

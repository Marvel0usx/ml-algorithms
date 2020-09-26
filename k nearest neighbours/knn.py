from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import numpy as np
import pandas as pd

FAKE_HEADER = r"./data/clean_fake.txt"
REAL_HEADER = r"./data/clean_real.txt"

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.5
TEST_SPLIT = 0.5

K_START = 1
K_END = 20

training_X = None
validating_X = None
testing_X = None


def load_data():
    """Function that victories news headers."""
    fake_headers = open(FAKE_HEADER, "r").readlines()
    real_headers = open(REAL_HEADER, "r").readlines()

    fake_header_vec = CountVectorizer()
    real_headers_vec = CountVectorizer()

    fake_header_vec.fit_transform(fake_headers)
    real_headers_vec.fit_transform(real_headers)

    features = fake_header_vec.get_feature_names() +\
                real_headers_vec.get_feature_names()

    new_fake_header_vec = CountVectorizer(vocabulary=features)
    new_real_header_vec = CountVectorizer(vocabulary=features)

    fake_header_X = new_fake_header_vec.fit_transform(fake_headers)
    real_header_X = new_real_header_vec.fit_transform(real_headers)

    dataset_t = np.concatenate(
        np.zeros((-1, fake_header_X.shape[0])),
        np.ones((-1, real_header_X.shpae[0])),
        axis=1
    )
    dataset_X = np.concatenate(fake_header_X, real_header_X, axis=0)
    dataset = pd.DataFrame(np.concatenate(dataset_X, dataset_t, axis=1), columns=features + ["y"])

    training_X_mask = np.random.rand(dataset.shape[0]) < TRAIN_SPLIT
    training_X = dataset[training_X_mask]
    remained_X = dataset[~training_X_mask]

    test_validation_mask = np.random.rand(remained_X.shape[0]) < TEST_SPLIT
    testing_X = remained_X[test_validation_mask]
    validating_X = remained_X[~test_validation_mask]


def select_knn_model(metric=None):
    models = {}
    for k in range(K_START, K_END + 1):
        this = KNeighborsClassifier(n_neighbors=k, metric=[metric, "minkowski"][metric is None])
        this.fit(training_X[:, :-1], training_X[:, -1])
        models[k] = [this]
        models[k].append(this.score(training_X[:, :-1], training_X[:, -1]))
        models[k].append(this.score(validating_X[:, :-1], validating_X[:, -1]))

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


if __name__ == "__main__":
    load_data()
    best_model, models = select_knn_model()
    best_model_2, models_2 = select_knn_model("cosine")
    plot_errors(models).savefig("knn-minkowski.png")
    plot_errors(models_2).savefig("knn-cosine.png")

    import pickle
    pickle.dump(models, open("knn-minkowski.dat", "wb"))
    pickle.dump(models_2, open("knn-cosine.dat", "wb"))

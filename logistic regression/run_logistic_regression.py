from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.008,
        "weight_regularization": 0.,
        "num_iterations": 500
    }
    weights = np.zeros((M + 1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train_stat = {}
    valid_stat = {}

    for t in range(hyperparameters["num_iterations"]):
        _, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights -= hyperparameters["learning_rate"] * df
        ce_train, _ = evaluate(train_targets, y)
        ce_valid, _ = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
        train_stat[t] = ce_train
        valid_stat[t] = ce_valid

    plt.plot(train_stat.keys(), train_stat.values(), label="Train")
    plt.plot(valid_stat.keys(), valid_stat.values(), label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy")
    plt.legend()
    plt.show()

    test_inputs, test_targets = load_test()
    print(evaluate(test_targets, logistic_predict(weights, test_inputs)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    valid_inputs, valid_targets = load_valid()

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.23,
        "weight_regularization": 0.0,
        "num_iterations": 800
    }
    lambds = (0, 0.001, 0.01, 0.1, 1.0)

    stat = []
    for loader in (load_train, load_train_small):
        train_inputs, train_targets = loader()
        N, M = train_inputs.shape
        dataset = ["mnist", "mnist_small"][loader is load_train_small]
        for lambd in lambds:
            hyperparameters["lambd"] = lambd
            for run in range(5):
                weights = np.random.randn(M + 1, 1)
                for epoch in range(hyperparameters["num_iterations"]):
                    f, df, y = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                    weights -= hyperparameters["learning_rate"] * df
                    train_ce, train_cr = evaluate(train_targets, y)
                    valid_ce, valid_cr = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
                    stat.append([dataset, lambd, run, epoch, train_ce, train_cr, valid_ce, valid_cr])
    import pandas as pd
    df_stat = pd.DataFrame(data=stat, columns=("dataset", "lambda", "run", "epoch", "train CE", "train CR", "valid CE", "valid CR"))

    # print summarized cross-entropy, classification rate for different lambdas and different datasets.
    summary = pd.pivot_table(df_stat, values=["train CE", "train CR", "valid CE", "valid CR"], index=["dataset", "lambda"],
                   aggfunc=np.mean)
    print(summary)
    # produce plots
    our_run_choice = 3
    for dataset in ("mnist", "mnist_small"):
        for lambd in lambds:
            df_plot = df_stat[(df_stat.dataset == "mnist") & df_stat.run.eq(our_run_choice) & df_stat["lambda"].eq(lambd)][["epoch", "train CE", "valid CE"]]
            plt.clf()
            plt.plot(df_plot["epoch"], df_plot["train CE"], label="Training CE")
            plt.plot(df_plot["epoch"], df_plot["valid CE"], label="Validation CE")
            plt.xlabel("Epochs")
            plt.ylabel("Cross Entropy")
            plt.legend()
            plt.title(f"dataset: {dataset}, lambda: {lambd}")
            plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    # run_logistic_regression()
    run_pen_logistic_regression()

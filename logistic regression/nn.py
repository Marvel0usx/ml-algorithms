import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        # Initialize random weights between -1 and 1
        self.weights = 2 * np.random.random((3, 1)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(0, training_iterations):
            # Let's ask the nn what the output is for the given training inputs. S(sum(..))
            output = self.think(training_inputs)
            # This is the error!
            error = training_outputs - output
            # Compute the adjustment
            adjustment = np.dot(training_inputs.T, error * NeuralNetwork.sigmoid_derivative(output))
            self.weights += adjustment

    def think(self, inputs):
        return NeuralNetwork.sigmoid(np.dot(inputs, self.weights))


if __name__ == "__main__":
    print("nn1")
    nn1 = NeuralNetwork()

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 0]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    print("Start to train")
    print(nn1.weights)
    nn1.train(training_inputs, training_outputs, 100)
    print("End of training")
    print(nn1.weights)

    # Let's see accuracy:
    print("Let's query the network")
    print(f"what is your response to [0, 0, 1]? {nn1.think(np.array([0, 0, 1]))}")
    # print(nn1.think(np.array([1, 1, 1])))
    # print(nn1.think(np.array([1, 0, 1])))
    # print(nn1.think(np.array([0, 1, 0])))

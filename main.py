import pandas as pd
import numpy as np


class Neuron_Layer:
    def __init__(self, n_inputs, n_neurons, eta):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(n_inputs, n_neurons))
        # 1 x n_neurons matrix
        self.biases = np.ones((1, n_neurons))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs, targets):

            target = targets[i]
            #inputs [1x784] dot [784 x 10] + [1x10]
            y = np.dot(inputs, self.weights) + self.biases
            # activation function is sigmoid
            y = self.sigmoid(y)
            #  finds the predicted class index specifically.
            y = np.argmax(y)

            return y
    # def checkTruth(self, groundTruths, inputs):
    #     groundTruths[inputs

    def backward(self, targets, yK):
        for i in range(len(targets)):

#          calculate loss





# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)



def main():
    # read in the MNIST data:
    mnist_data = pd.read_csv("mnist_train.csv")
    # collect all of the ground_truths.
    groundTruthLables = mnist_data['label']
    print(groundTruthLables)
    # drop the labels of the data and then divide each array
    inputs = mnist_data.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = np.divide(inputs, 255)
    print(normalizedInputs.shape)
    # obtain the shape for each input to consistently maintain dotProduct
    # normalizedInputs = normalizedInputs.reshape((-1, 28, 28))
    # confirms 28 x 28
    # print(normalizedInputs.shape)
    # print(normalizedInputs[0])
    dense1 = Neuron_Layer(784, 10,0.1)
    print(dense1.forward(normalizedInputs,groundTruthLables))



main()

import pandas as pd
import numpy as np


# imports
def importMNIST():
    mnist_data = pd.read_csv("mnist_train.csv")
    # collect all of the ground_truths.
    groundTruthLables = mnist_data['label']
    print(groundTruthLables)
    # drop the labels of the data and then divide each array
    inputs = mnist_data.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = np.divide(inputs, 255)
    # obtain the shape for each input to consistently maintain dotProduct
    normalizedInputs = normalizedInputs.reshape((-1,28,28))
    # confirms 28 x 28
    print(normalizedInputs.shape)

    print(normalizedInputs[0])


    # need to normalize the data. which is gray scale images of numbers 1-9




# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)

# 785 weights per perceptron




def main():
    # read in the MNIST data:
    importMNIST()

    ## define bias:
    bias = 1



main()


import pandas as pd
import numpy as np


class Perceptron_Layer:
    def __init__(self, n_inputs, n_neurons, learning_rate):
        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(n_inputs, n_neurons))
        # 1 x n_neurons matrix
        self.biases = np.ones((1, n_neurons))
        self.eta = learning_rate


    #  takes a single input
    def forward(self, input, target, adjustment):

        #  convert the target to a vector representation



        #inputs [1x784] dot [784 x 10] + [1x10]
        activationPotential = np.dot(input, self.weights)
        # perceptron results
        # [1 x 10] result matrix

        prediction = np.argmax(activationPotential)
        if target == prediction:
            self.accurateCount += 1
        else:
            self.incorrect += 1

            #  for HW1 algo... single batch input.. get the single array input and or access weights for tuning...
            if adjustment:
                #  threshold check
                #  contains the binary classification from potential fire
                activationPotential = np.where(activationPotential > 0, 1, 0)
                #  create a truthVector t^i per equation for weight adjustment
                truthVector = np.zeros(10, dtype="int")

                if activationPotential[target] == 1:
                    #  set that single index to 1 if correct prediction
                    truthVector[target] = 1

                #  subtract for inner (t^i - y^i)
                innerDelta = truthVector - activationPotential

                self.weights = self.weights + self.eta * np.outer(innerDelta, input).T

                #  update weights
        #         for weightRow in range(len(self.weights)):
        #             for weightIndex in range(len(self.weights[weightRow])):
        #                 # calculate new weight for assignment
        #
        #                 for perceptronResp in range(len(activationPotential)):
        #                     self.weights[weightRow, weightIndex] = self.weights[weightIndex][perceptronResp] + self.eta * ((truthVector[perceptronResp]) - activationPotential[perceptronResp]) * input[weightRow]
        # #          iterate each feature for the corresponding weights.
        #  weights [784 x 10] features per input [1 x 784]


        # count this iteration:
        self.inputSize += 1


    def determinePerformance(self):
        self.accuracy = round(((self.accurateCount / self.inputSize)*100), 2)
        print("Accuracy : " + str(self.accuracy))

    def clearAccuracy(self):
        self.accuracy = 0
        self.accurateCount = 0
        self.inputSize = 0




# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)

def main():
    # read in the MNIST data:
    mnist_data = pd.read_csv("mnist_train.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistData = mnist_data.sample(frac=1).reset_index(drop=True)
    # collect all of the ground_truths.
    groundTruthLables = shuffledMnistData['label'].to_numpy()
    # drop the labels of the data and then divide each array
    inputs = mnist_data.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = inputs / 255
    # obtain the shape for each input to consistently maintain dotProduct
    # normalizedInputs = normalizedInputs.reshape((-1, 28, 28))
    # confirms 28 x 28
    # print(normalizedInputs.shape)
    # print(normalizedInputs[0])
    dense1 = Perceptron_Layer(784, 10,0.1)

    #  single input iteration for the batch size of training being 1
    #  initial epoch

    #  initial test run...
    print("Epoch: 0")
    print('=============================')
    for d in range(len(normalizedInputs)):
        digit = normalizedInputs[d]
        truth = groundTruthLables[d]
        dense1.forward(digit, truth, False)
    dense1.determinePerformance()
    dense1.clearAccuracy()

    epochCount = 0
    print("Epoch" + str(epochCount) + ": learning rate 0.1")
    print('=============================')
    for d in range(60000):
        digit = normalizedInputs[d]
        truth = groundTruthLables[d]
        dense1.forward(digit, truth, True)
        dense1.determinePerformance()
        epochCount += 1



    dense1.determinePerformance()


main()

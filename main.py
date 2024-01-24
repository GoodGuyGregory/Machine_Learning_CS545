import pandas as pd
import numpy as np


# imports
def importMNIST():
    MNIST = pd.read_csv("mnist_train.csv", header=None)
    MNIST.head()



def main():
    # read in the MNIST data:
    importMNIST()



main()


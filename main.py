import numpy as np
from matplotlib import pyplot
from keras.datasets import cifar10

def main():
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

if __name__ == "__main__":
    main()
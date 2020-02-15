import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from readData import readPinchFromNP

def main():

    preprocessedData = readPinchFromNP()

    labels = preprocessedData[0] # ((n, 5)) numpy array  of labels
    data = preprocessedData[1] # ((n, 8)) numpy array of averages of every session


    print(labels.shape)
    print(data.shape)

    #now we can start fucking with keras


if __name__ == "__main__":
    main()




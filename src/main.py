import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from readData import readPinchFromNP
from formats import flattenForAverage

def main():
    # so far this only processes session 1, to have it process all sessions
    # either change readData.py or run this 23 times
    
    unprocessedData = readPinchFromNP()
    preprocessedData = flattenForAverage(x[0], x[1])

    labels = preprocessedData[0] # ((n, 5)) numpy array  of labels
    data = preprocessedData[1] # ((n, 8)) numpy array of averages of every session

    #now we can start fucking with keras


if __name__ == "__main__":
    main()




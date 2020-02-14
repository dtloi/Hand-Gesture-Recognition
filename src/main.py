import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from readData import readPinchFromNP
from formats import flattenForAverage

def main():
    x = readPinchFromNP()
    print("finished Process1")
    flattenForAverage(x[1], x[0])




if __name__ == "__main__":
    main()




import numpy as np
import matplotlib.pyplot as plt
import warnings

# Uncomment for running on Windows
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM

# Uncomment for running on Mac
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM

from readData import readPinchFromNP
from formats import flattenForAverage, lstm

warnings.filterwarnings('ignore') #we'll probably need to turn this back on

def trylstm():
    dataset = np.load('./temp/tripled.npy')
    labels = np.load('./temp/tripled_label.npy')
    #print(dataset.shape)
    #print(labels.shape)
    #print(dataset)
    #print(labels)
    train_labels, test_labels = np.split(labels, [800, ])
    train_data, test_data = np.split(dataset, [800, ])
    
    model = Sequential()
    model.add(LSTM(units = 7, return_sequences = True, input_shape = (train_data.shape[1], 8)))
    model.add(LSTM(units = 6, return_sequences = True))

    model.add(Dense(units = 5))
    model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics =['acc'])
    history = model.fit(train_data, train_labels, epochs = 5, batch_size = 50, validation_split=0.2)
    
    train_loss, train_acc = model.evaluate(train_data, train_labels)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Training set accuracy:', train_acc)
    print('Test set accuracy:', test_acc)

    # The history of our accuracy during training.
    model.summary()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



    # The history of our cross-entropy loss during training.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Number of epochs')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


    plot_decision_boundary(test_data, test_labels, model) 
    # Reset figure size back to default.
    rcParams['figure.figsize'] = 6, 4

    #
    # This code is substantively from https://rohitmidha23.github.io/Neural-Network-Decision-Boundary/
    #

def main():
    preprocessedData = readPinchFromNP()
    labels = preprocessedData[0] # ((n, 5)) numpy array  of labels
    data = preprocessedData[1] # ((n, 8)) numpy array of averages of every session

    lstm(labels, data)

    trylstm()


    # preprocessedData = readPinchFromNP()

    # labels = preprocessedData[0] # ((n, 5)) numpy array  of labels
    # data = preprocessedData[1] # ((n, 8)) numpy array of averages of every session

    # train_labels, test_labels = np.split(labels, [2000, ])
    # train_data, test_data = np.split(data, [2000, ])

    # print(train_labels.shape)
    # print(test_labels.shape)
    # print(train_data.shape)
    # print(test_data.shape)

    # input = Input(shape=(8,))
    # x1 = Dense(128, activation='relu')(input)
    # x2 = Dense(64, activation='relu')(x1)
    # x3 = Dense(32, activation='relu')(x2)
    # x4 = Dense(16, activation='relu')(x3)
    # x5 = Dense(8, activation='relu')(x4)
    # x6 = Dense(5, activation='sigmoid')(x5)

    # model = Model(inputs=input, outputs=x6)

    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

    # history = model.fit(train_data, train_labels, validation_split=0.1, epochs=150, batch_size=32)

    # train_loss, train_acc = model.evaluate(train_data, train_labels)
    # test_loss, test_acc = model.evaluate(test_data, test_labels)

    # print('Training set accuracy:', train_acc)
    # print('Test set accuracy:', test_acc)

    # model.summary()

    # # The history of our accuracy during training.
    # print(history.history.keys())
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of epochs')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # # The history of our cross-entropy loss during training.
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Number of epochs')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()


if __name__ == "__main__":
    main()




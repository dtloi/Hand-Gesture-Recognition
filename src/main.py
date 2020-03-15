import numpy as np
import matplotlib.pyplot as plt
import warnings

# Uncomment for running on Windows
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM

# Uncomment for running on Mac
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.utils import plot_model

from readData import readPinchFromNP, readRoshamboFromNP
from formats import lstm

warnings.filterwarnings('ignore') #we'll probably need to turn this back on

def trylstm():
    dataset = np.load('./temp/tripled.npy')
    labels = np.load('./temp/tripled_label.npy')
    #print(dataset.shape)
    #print(labels.shape)
    #print(dataset)
    #print(labels)
    train_labels, test_labels = np.split(labels, [2478, ])
    train_data, test_data = np.split(dataset, [2478, ])
    
    model = Sequential()
    model.add(LSTM(units = 8, return_sequences = True, input_shape = (train_data.shape[1], 8), kernel_initializer='random_uniform',))
    model.add(LSTM(units = 64, return_sequences = True))
    model.add(LSTM(units = 32, return_sequences = True))
    model.add(LSTM(units = 16, return_sequences = True))
    model.add(Dense(units = 8))
    model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics =['acc'])

    
    history = model.fit(train_data, train_labels, epochs = 15, batch_size = 32, validation_split=0.2)
   
    
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

    plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        dpi=300)
    return


def main():


    preprocessedPinchData = readPinchFromNP()
    pinchLabels = preprocessedPinchData[0] # ((n, 5)) numpy array  of labels
    pinchData = preprocessedPinchData[1] # ((n, 8)) numpy array of averages of every session

    preprocessedRoshamboData = readRoshamboFromNP()
    roshamboLabels = preprocessedRoshamboData[0] # ((n, 5)) numpy array  of labels
    roshamboData = preprocessedRoshamboData[1] # ((n, 8)) numpy array of averages of every session
    print(roshamboData.shape)

    # i = 1
    # size = 270177
    # count = 0
    # min = 240
    # while (i < size):
    #     if roshamboLabels[i] != roshamboLabels[i-1]:
    #         #print(count)
    #         if count < min:
    #             min = count
    #         count = 0
    #     count += 1
    #     i += 1
    # print(min)
    
    labels = np.append(pinchLabels, roshamboLabels, axis = 0)
    data = np.append(pinchData, roshamboData, axis = 0)

    #you can comment this out if you already have the formatted data
    #lstm(labels, data) #uncomment this if you want to reformat the data. This takes a while...
    trylstm()

if __name__ == "__main__":
    main()




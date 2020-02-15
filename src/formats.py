import numpy as np

#params:
# data - list of 1x8 arrays containing sensor data from one session
# labels - list of labels containing label names corrosponding to each element in data
# this function doesnt work, need to mess with indexes...
def flattenForAverage(labels, data):
    segments = []
    flattenedLabels = []
    i = 1 #start at second element
    #print(len(labels))
    while(i < len(labels)):
        if not(labels[i] == labels[i-1]):
            segments.append(i-1)
            flattenedLabels.append(labels[i-1])
        i+=1
        #print(i)
    #print(flattenedLabels) # turn this into a numpy array
    
    dataArray = np.zeros((1,8))
    i = 0
    while(i < len(segments)):
        if i == 0:
            start = 0
        else:
            start = segments[i-1] + 1

        size = segments[i]
        averagedSensors = np.zeros((1,8))
        
        while(start < size):
            averagedSensors = averagedSensors + data[start]
            start+=1
        averagedSensors = averagedSensors / (size - segments[i-1])
        #print(size)
        dataArray = np.append(dataArray, averagedSensors, axis=0)
        i+=1
    dataArray = np.delete(dataArray, 0, axis=0)
    #print(dataArray)

    # at this point, the data just needs a bias column added
    # now we will set the labels
    labelArray = np.zeros((1, 5))
    for label in flattenedLabels:
        #print(labelArray.shape)
        #print(labelSwitch(label))
        labelArray = np.append(labelArray, labelSwitch(label), axis=0)
    labelArray = np.delete(labelArray, 0, axis=0)
    #print(labelArray)

    return [labelArray, dataArray]


def pinchOne():
    return np.array([[1, 0, 0, 0, 0]])
def pinchTwo():
    return np.array([[0, 1, 0, 0, 0]])
def pinchThree():
    return np.array([[0, 0, 1, 0, 0]])
def pinchFour():
    return np.array([[0, 0, 0, 1, 0]])
def none():
    return np.array([[0, 0, 0, 0, 1]])

def labelSwitch(label):
    switch = {
        'Pinch1': pinchOne,
        'Pinch2': pinchTwo,
        'Pinch3': pinchThree,
        'Pinch5': pinchFour,
        'none': none
    }
    func = switch.get(label, lambda: np.array([[0, 0, 0, 0, 0]]))
    return func()


        
            
        


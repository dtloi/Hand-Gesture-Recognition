import numpy as np

#params:
# data - list of 1x8 arrays containing sensor data from one session
# labels - list of labels containing label names corrosponding to each element in data
# this function doesnt work, need to mess with indexes...
def flattenForAverage(data, labels):
    segments = []
    flattenedLabels = []
    i = 1 #start at second element
    print(len(labels))
    while(i < len(labels)):
        if not(labels[i] == labels[i-1]):
            segments.append(i-1)
            flattenedLabels.append(labels[i-1])
        i+=1
        #print(i)
    print(flattenedLabels) # turn this into a numpy array
    
    totaledSensors = np.zeros((1,8))
    for size in segments:
        averagedSensors = np.zeros((1,8))
        j = 0
        while(j < size):
            averagedSensors = averagedSensors + data[j]
            j+=1
        #averagedSensors = averagedSensors / size
        print(size)
        totaledSensors = np.append(totaledSensors, averagedSensors, axis=0)
    totaledSensors = np.delete(totaledSensors, 0, axis=0)
    print(totaledSensors)
        
            
        


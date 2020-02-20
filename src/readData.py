import numpy as np

'''
NOTE: This loop only works for the Pinch dataset. 
If you want to create a loop for the Roshambo dataset, just change the folder name and the loop parameters.

'''

def readPinchFromNP():
    #file = open("first_session.txt", "w") # remove this later, only for testing
    #declare np array with random values because there were some issues with getting an empty np array
    masterLabels = np.zeros((1,))
    masterData = np.zeros((1, 8))
    #fin_res = pd.DataFrame({})
    for subject in range(1, 23): #there's a bug when you try to enable all 22 subjects
        for session in range(1, 4): # for now we only want to look at one subject...
            #building the file name
            f_name = "../Pinch/subject"
            if (subject < 10):
                f_name = f_name + "0" + str(subject)
            else:
                f_name = f_name + str(subject)
            f_name = f_name + "_session"
            if (session < 10):
                f_name = f_name + "0" + str(session)
            else:
                f_name = f_name + str(session)
            f_name_ann = f_name + "_ann.npy"
            f_name_emg = f_name + "_emg.npy"

            #loading the file
            labels = np.load(f_name_ann)
            data = np.load(f_name_emg)
            #print(masterLabels.shape)
            #print(labels.shape)
            #print(data.shape)
            masterLabels = np.append(masterLabels, labels, axis=0)
            masterData = np.append(masterData, data, axis=0)
            
            
            
            #print(masterData)
            # concatonating results and data
            # result = pd.concat([df_emg, df_ann], axis=1).reindex(df_ann.index)
            # result.columns = [0,1,2,3,4,5,6,7,8]
            # print(result.to_string())
            # file.write(result.to_string())
            # fin_res.append(result)
            # print(result.loc[result[8] == 'Pinch2'])

            #exit()
    #print(fin_res)
    #file.close()

    #delete initial empty array 
    masterLabels = np.delete(masterLabels, 0, axis=0)
    masterData = np.delete(masterData, 0, axis=0)
    return [masterLabels, masterData]

def readRoshamboFromNP():
    #file = open("first_session.txt", "w") # remove this later, only for testing
    #declare np array with random values because there were some issues with getting an empty np array
    masterLabels = np.zeros((1,))
    masterData = np.zeros((1, 8))
    #fin_res = pd.DataFrame({})
    for subject in range(1, 11): #there's a bug when you try to enable all 22 subjects
        for session in range(1, 4): # for now we only want to look at one subject...
            #building the file name
            f_name = "../Roshambo/subject"
            if (subject < 10):
                f_name = f_name + "0" + str(subject)
            else:
                f_name = f_name + str(subject)
            f_name = f_name + "_session"
            if (session < 10):
                f_name = f_name + "0" + str(session)
            else:
                f_name = f_name + str(session)
            f_name_ann = f_name + "_ann.npy"
            f_name_emg = f_name + "_emg.npy"

            #loading the file
            labels = np.load(f_name_ann)
            data = np.load(f_name_emg)
            #print(masterLabels.shape)
            #print(labels.shape)
            #print(data.shape)
            masterLabels = np.append(masterLabels, labels, axis=0)
            masterData = np.append(masterData, data, axis=0)
            
            
            
            #print(masterData)
            # concatonating results and data
            # result = pd.concat([df_emg, df_ann], axis=1).reindex(df_ann.index)
            # result.columns = [0,1,2,3,4,5,6,7,8]
            # print(result.to_string())
            # file.write(result.to_string())
            # fin_res.append(result)
            # print(result.loc[result[8] == 'Pinch2'])

            #exit()
    #print(fin_res)
    #file.close()

    #delete initial empty array 
    masterLabels = np.delete(masterLabels, 0, axis=0)
    masterData = np.delete(masterData, 0, axis=0)
    return [masterLabels, masterData]


    

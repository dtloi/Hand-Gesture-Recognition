import numpy as np
import pandas as pd
'''
NOTE: This loop only works for the Pinch dataset. 
If you want to create a loop for the Roshambo dataset, just change the folder name and the loop parameters.

'''

#params: none
#return: np_array touples -> 0 is labels, 1 is data points...
def readPinchFromNP():
    file = open("first_session.txt", "w") # remove this later, only for testing

    fin_res = pd.DataFrame({})
    for subject in range(1, 2):
        for session in range(1, 2): # for now we only want to look at one subject...
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
            data_ann = np.load(f_name_ann)
            data_emg = np.load(f_name_emg)
            df_ann = pd.DataFrame(data_ann)
            df_emg = pd.DataFrame(data_emg)

            #concatonating results and data
            result = pd.concat([df_emg, df_ann], axis=1).reindex(df_ann.index)
            result.columns = [0,1,2,3,4,5,6,7,8]
            #print(result.to_string())
            file.write(result.to_string())
            fin_res.append(result)
            #print(result.loc[result[8] == 'Pinch2'])

            #exit() /1
    #print(fin_res)
    file.close()
    return [data_ann, data_emg]


'''
for subject in range(1, 11):
  for session in range(1, 4):
    #building the file name
    f_name = "/content/drive/My Drive/Roshambo/subject"
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

    #loding the file
    data_ann = np.load(f_name_ann)
    data_emg = np.load(f_name_emg)
    df_ann = pd.DataFrame(data_ann)
    df_emg = pd.DataFrame(data_emg)

    #concatonating results and data
    result = pd.concat([df_emg, df_ann], axis=1).reindex(df_ann.index)
    result.columns = [0,1,2,3,4,5,6,7,8]
    print(result)
#np.savetxt("foo.csv", data, delimiter=",")
'''

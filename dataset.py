import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import shuffle

# Global
numFiles = 30
endRow = 70000

# Depression data
dep_data=[]

for fileNum in range(1,numFiles+1):
    fileNum = f"{fileNum:02d}"
    location = 'modified_data/depression-mat/leftclosed'+ fileNum +'.txt'
    df = np.loadtxt(location)
    df=  df[0:endRow]
    dep_data.append(df)
    
dep_data = stats.zscore(dep_data)
dep_data = np.asarray(dep_data)
dep_data = np.reshape(dep_data,[700, 3000])

# Normal data
nor_data=[]

for fileNum in range(1,numFiles+1):
    fileNum = f"{fileNum:02d}"
    location = 'modified_data/depression-mat/leftclosed'+ fileNum +'.txt'
    df = np.loadtxt(location)
    df=  df[0:endRow]
    nor_data.append(df)
    
nor_data = stats.zscore(nor_data)
nor_data = np.asarray(nor_data)
nor_data = np.reshape(nor_data,[700, 3000])

# Assembling the data
data = np.concatenate((dep_data,nor_data),axis=1)
data = np.transpose(data)

# Building labels
label = np.concatenate((np.ones(3000),np.zeros(3000)), axis=0)

#shuffling the set
data,label =shuffle(data, label, random_state=0) 

#reshaping the data
data= data.reshape(data.shape[0], data.shape[1], 1)

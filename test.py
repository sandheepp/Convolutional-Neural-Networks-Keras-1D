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

#Reshaping the data
#label= pd.Categorical(label)
#data = np.reshape(data, [6000,1, 700, 1])
data= data.reshape(data.shape[0], data.shape[1], 1 )
#label = np.reshape(label,[1, label.shape])
#label = np.reshape(label,[6000])

#Training Data with label
#data_with_label = np.vstack([data, label])

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

#K.set_image_dim_ordering('tf')
#keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

#CNN network
model = Sequential()
model.add(Conv1D(filters= 32, kernel_size = 5, input_shape=(700,1)))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='softmax'))
#model.add(Activation('softmax'))

#Optimizers
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd')

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)

#training
model.fit(X_train, y_train, batch_size=10, epochs=1)

#evaluate
score = model.evaluate(X_test, y_test, batch_size=15)
print(score)


#Prediction
predictions = model.predict(X_test, batch_size=None, verbose=0)






# Prediction-No
# a=np.expand_dims(dataset[23000], axis=0)
# output=model.predict(a)
# print(output)

#Random prediction
# x_input = data[6]
# x_input = x_input.reshape((1,700, 1))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
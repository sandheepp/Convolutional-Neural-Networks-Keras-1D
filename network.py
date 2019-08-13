#Import
from dataset import data, label

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

#CNN network
model = Sequential()
model.add(Conv1D(filters= 32, kernel_size = 5, input_shape=(700,1)))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.add(Activation('softmax'))

#Optimizers
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)

#training
model.fit(X_train, y_train, batch_size=10, epochs=1)

#evaluate
score = model.evaluate(X_test, y_test, batch_size=10)
print(score)

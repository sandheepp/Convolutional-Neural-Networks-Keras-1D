import time
t = time.time()

#Import training data and labels
from dataset import data, label

# Count elapsed time and print it
elapsed = time.time() - t
print("Time taken for data import(in sec):"+str(elapsed))

#importing modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers, metrics
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

# Count elapsed time for import and print it
elapsed = time.time() - t
print("Time taken for data/modules import(in sec):"+str(elapsed))


#input data shape
data_shape=data.shape[1] 

#CNN network
model = Sequential()
model.add(Conv1D(filters= 32, kernel_size = 5, input_shape=(data_shape,1)))
model.add(MaxPooling1D(pool_size=2,strides=2))
#model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Activation('softmax'))

#Optimizers
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

#Configures the model for training.
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#Data Splitting: Test, train
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=0)

# Train to train, CV
X_train, X_CV, y_train, y_CV = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

#Training : Trains the model for a given number of epochs (iterations on a dataset).
model.fit(X_train, y_train, batch_size=64, epochs=1)

#evaluate: Returns the loss value & metrics values(accuracy) for the model in test mode.
score = model.evaluate(X_CV, y_CV, batch_size=10)
print(score)

#Generates output predictions for the input samples.
predictions = model.predict(X_test, batch_size=None, verbose=0)
print(predictions)

#abcd
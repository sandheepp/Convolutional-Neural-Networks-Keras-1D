import time
t = time.time()

#Import training data and labels from the other python file
from dataset import data, label

# Count elapsed time and print it
elapsed = time.time() - t
print("Time taken for data import from dataset.py(in sec):"+str(elapsed))

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


# # RELU function with custom arguments
# def relu(x, alpha=1000, max_value=0.50, threshold=0.50):
#     """Rectified Linear Unit.
#     With default values, it returns element-wise `max(x, 0)`.
#     Otherwise, it follows:
#     `f(x) = max_value` for `x >= max_value`,
#     `f(x) = x` for `threshold <= x < max_value`,
#     `f(x) = alpha * (x - threshold)` otherwise.
#     # Arguments
#         x: Input tensor.
#         alpha: float. Slope of the negative part. Defaults to zero.
#         max_value: float. Saturation threshold.
#         threshold: float. Threshold value for thresholded activation.
#     # Returns
#         A tensor.
#     """
#     return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)

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
#model.add(Activation("relu"))

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
print("Evaluation score is ", score)

#Generates output predictions for the input samples.
predictions = model.predict(X_test, batch_size=None, verbose=0)

#predictions, y_test
v = np.empty([600, 2]) 
v[:,0] = predictions[:,0]
v[:,1]= y_test[:] 

#print(v)

#save the value to a text file
np.savetxt("a.txt",v,fmt='%4f')

#Plotting the predictions and test in the same graph

# # importing the required module 
# import matplotlib.pyplot as plt 

# # line 1 points 
# x1 = list(range(1, 601)) 
# y1 = v[:,0]
# # plotting the line 1 points  
# plt.plot(x1, y1, label = "predictions") 
  
# # line 2 points 
# x2 =list(range(1, 601)) 
# y2 =  v[:,1]
# # plotting the line 2 points  
# plt.plot(x2, y2, label = "y_test") 
  
# # naming the x axis 
# plt.xlabel('x - axis') 
# # naming the y axis 
# plt.ylabel('y - axis') 
# # giving a title to my graph 
# plt.title('Two lines on same graph!') 
  
# # show a legend on the plot 
# plt.legend() 
  
# # uncomment the function to show the plot 
# plt.show() 

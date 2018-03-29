import keras
from keras.models import load_model
import os
import time
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

CNN_model = load_model('CNN_model_trained_v1.model')


X_test = np.load('X_test.npy')
X_test = X_test / 255
Y_test = np.load('Y_test.npy')



Y_pred = CNN_model.predict(X_test)
Y_pred = Y_pred >= 0.5
Y_pred = 1*Y_pred

print (confusion_matrix(Y_test,Y_pred))



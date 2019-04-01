import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import random
import time

from sklearn import preprocessing, model_selection
import tensorflow as tf

#from tf.keras.models import Sequential
#from tf.keras.layers.Dense import Dense
#from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
#from tf.keras.utils.to_categorical import to_categorical
from sklearn.utils import shuffle

data = pd.read_csv('Iris.csv')
data = data.drop(['Id'], axis =1)

data = shuffle(data)

i = 8
data_to_predict = data[:i].reset_index(drop = True)
predict_species = data_to_predict.Species
predict_species = np.array(predict_species)
prediction = np.array(data_to_predict.drop(['Species'],axis= 1))

data = data[i:].reset_index(drop = True)

X = data.drop(['Species'], axis = 1)
X = np.array(X)
Y = data['Species']

#Transform name species into numerical values
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = tf.keras.utils.to_categorical(Y)
#print(Y)

#We have 3 classes : the output looks like :
#0,0,1 : Class 1
#0,1,0 : Class 2
#1,0,0 : Class 3

train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)

input_dim = len(data.columns) - 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim = input_dim , activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'relu'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 100, batch_size = 8)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict_classes(prediction)
prediction_ = np.argmax( tf.keras.utils.to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)

for i, j in zip(prediction_ , predict_species):
    print( " the nn predict {}, and the species to find is {}".format(i,j))
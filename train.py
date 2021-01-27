# MOBILE PRICE RANGE CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 
# VIGNESHWAR RAVICHANDAR

#IMPORTING MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model 
import matplotlib.pyplot as plt

# IMPORTING DATA
data = pd.read_csv("mobile_data.csv")

# PREPROCESSING DATA
x = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
y = data.iloc[:,12].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)

# BUILDING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(64, input_dim = 12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

# TRAINING THE DATA
opt = Adam(learning_rate = 0.01)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 150, batch_size = 32, validation_data = (x_val, y_val))
print("\n\n ----- Model is trained successfully ! ----- \n\n")



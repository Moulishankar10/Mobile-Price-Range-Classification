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

# IMPORTING MODULES
data = pd.read_csv("mobile_data.csv")

# PREPROCESSING DATA
x = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
y = data.iloc[:,12].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)




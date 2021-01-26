# MOBILE PRICE RANGE CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 

#IMPORTING MODULES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

# IMPORTING MODULES
data = pd.read_csv("data/data.csv")

# PREPROCESSING DATA
x = data.iloc[:,[:-2]].values
y = data.iloc[:,-1].values

# SPLITTING THE TRAINING AND VALIDATION DATA
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state = 0)


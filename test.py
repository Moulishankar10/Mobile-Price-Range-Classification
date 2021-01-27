# MOBILE PRICE RANGE CLASSIFICATION

# DEVELOPED BY
# MOULISHANKAR M R 
# VIGNESHWAR RAVICHANDAR

#IMPORTING MODULES
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# LOADING THE TRAINED MODEL
model = load_model("model/model",custom_objects=None,compile=True)

# INPUT DATA
print("\n\n Enter the following specifications of a mobile phone to explore its price range. \n\n")

bc = int(input("\nBattery Capacity (mAh) : "))
ds = input("\nDual SIM (y/n) : ")
fc = int(input("\nFront Camera (mega pixels) : "))
rc = int(input("\nRear Camera (mega pixels) : "))
im = int(input("\nInternal Memory (GB) : "))
fs = input("\nFingerprint Sensor (y/n) : ")
ram = int(input("\nRAM (GB): "))
pb = int(input("\nProcessor Benchmark : "))
dw = int(input("\nDisplay Width (pixel) : "))
dl = int(input("\nDisplay Length (pixel) : "))
fg = input("\n5G (y/n) : ")
fcs = input("\nFast Charging Support (y/n) : ")

# PROCESSING INPUT DATA
ds = 1 if ds == 'y' else 0
fs = 1 if fs == 'y' else 0
fg = 1 if fg == 'y' else 0
fcs = 1 if fcs == 'y' else 0

spec = [bc,ds,fc,rc,im,fs,ram,pb,dw,dl,fg,fcs]

# DEFINING CLASSIFICATIONS
op1 = ["Below Rs.10,000","Rs.10,000 - Rs.20,000","Rs.20,000 - Rs.30,000","Above Rs.30,000"]
op2 = ["Low Cost","Medium Cost","High Cost","Very High Cost"]
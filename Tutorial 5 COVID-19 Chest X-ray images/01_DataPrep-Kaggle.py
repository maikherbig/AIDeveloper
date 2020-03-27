import os
import numpy as np
import pandas as pd
from shutil import copyfile
from keras.preprocessing.image import load_img
import PIL
color_mode = "grayscale"

classes = ["NORMAL","PNEUMONIA"]
for class_ in classes:
    folder_orig =  "."+os.sep+class_
    folder_resiszed = "."+os.sep+class_+"_res"
    if not os.path.isdir(folder_resiszed):#check if a folder called "train_2" exists
        os.mkdir(folder_resiszed)##if not, create one
    
    files = os.listdir(class_)
    for file in files:
        img = load_img(folder_orig+os.sep+file,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
        img = img.resize((360,360),resample=PIL.Image.LANCZOS)
        img.save(folder_resiszed+os.sep+file)
        print(file)













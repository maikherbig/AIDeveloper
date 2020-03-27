import os
import numpy as np
import pandas as pd
from shutil import copyfile
from keras.preprocessing.image import load_img
import PIL

#Insert the path the directory containing the files
direc = r"D:\BIOTEC-WORK\Own ideas\62 COVID-19 X ray\Covid_Git_Xray\covid-chestxray-dataset-master"
os.chdir(direc) #change the working directory

color_mode = "grayscale"

Table = pd.read_csv("metadata.csv")
Table = Table[Table["modality"]=="X-ray"]#keep only X-ray acquired frontally
keys = list(Table.keys())

folder_orig =  "."+os.sep+"images"
folder_resiszed = "."+os.sep+"images_resized2"
if not os.path.isdir(folder_resiszed):#check if a folder called "train_2" exists
    os.mkdir(folder_resiszed)##if not, create one

diagnoses = Table["finding"].unique()
for diagnosis in diagnoses:
    folder_diag = folder_resiszed+os.sep+diagnosis
    if not os.path.isdir(folder_diag):#check if a folder called "train_2" exists
        os.mkdir(folder_diag)##if not, create one

    Table_temp = Table[Table["finding"]==diagnosis]#keep only X-ray acquired frontally
    view  = Table_temp["view"]
    ind = ["PA" in v or "AP" in v for v in view]
    Table_temp = Table_temp[ind].reset_index(drop=True)
    filename = Table_temp["filename"]
    for filen in filename:
        print(filen)
        img = load_img(folder_orig+os.sep+filen,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
        img = img.resize((360,360),resample=PIL.Image.LANCZOS)
        img.save(folder_diag+os.sep+filen)













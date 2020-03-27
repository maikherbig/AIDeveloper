import os
import numpy as np
import pandas as pd
from shutil import copyfile
import PIL
from keras.preprocessing.image import load_img
color_mode = "grayscale"
tr_or_valid = "valid"
Table = pd.read_csv(tr_or_valid+".csv")
Table = Table[Table["Frontal/Lateral"]=="Frontal"]
keys = list(Table.keys())

#classes = keys[5:17] #different diagnoses, excluding fracture
classes = ["No Finding","Pneumonia"]
#create new folder for training/validation
if not os.path.isdir(tr_or_valid+"_2"):
    os.mkdir(tr_or_valid+"_2")

for class_ in classes:
    Table_diag = Table[Table[class_]==1]
    foldername = class_.replace(" ","_")
    #Create folder for each Diagnosis
    if not os.path.isdir(tr_or_valid+"_2"+os.sep+foldername):
        os.mkdir(tr_or_valid+"_2"+os.sep+foldername)
    
    #get the directories of all the images of this class
    paths = Table_diag["Path"]
    paths = [p.split("CheXpert-v1.0-small")[1][1:] for p in paths]
    
    for path in paths:
        try:
            img = load_img(path,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
            img = img.resize((360,360),resample=PIL.Image.LANCZOS)
            img.save(tr_or_valid+"_2"+os.sep+foldername+os.sep+path.split(tr_or_valid)[1][1:].replace("/","_"))
            print(tr_or_valid+"_2"+os.sep+foldername+os.sep+path.split(tr_or_valid)[1][1:].replace("/","_"))
        except:
            pass















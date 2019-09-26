import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.preprocessing.image import load_img

color_mode = "Grayscale"
#in Fashion MNIST are the following classes:
classes = {0:"Top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Boot"}

for dataset in ["test","train"]:
    fname = "fashion-mnist_"+dataset+".csv"
    csv = pd.read_csv(fname)
    os.mkdir(dataset) #create "test" or "train" directory
    for col in classes.keys():
        #start with first class
        os.mkdir(dataset+os.sep+str(col)+"_"+classes[col]) #create directory for the resepctive class
        #search in csv table for class 0:
        label = csv["label"].values
        ind = np.where(label==col)[0]
        images = csv.loc[ind]
        images = images.drop(columns="label").values
        #go through, line by line (each line is an image)
        for row in range(images.shape[0]):
            image = images[row,:]
            image = image.reshape(28,28) #reshape to 28x28
            image_pil = Image.fromarray(image.astype(np.uint8))
            image_pil.save(dataset+os.sep+str(col)+"_"+classes[col]+os.sep+str(col)+"_"+classes[col]+"_"+str(row)+".PNG")


def test_csv_and_png_equal():
    #test if png data is still exactly the same as csv data
    #Grab a testing png
    path = os.path.join("test","0_Top","0_Top_0.PNG")
    image_png = load_img(path,color_mode=color_mode.lower()) #This uses PIL and supports many many formats!
    image_png = np.array(image_png)
    
    #Get the same image in csv file
    csv = pd.read_csv("fashion-mnist_test.csv")
    label = csv["label"].values
    ind = np.where(label==0)[0]
    images = csv.loc[ind]
    images = images.drop(columns="label").values
    image_csv = images[0,:]
    image_csv = image_csv.reshape(28,28)
    
    assert np.allclose(image_csv,image_png)

test_csv_and_png_equal()

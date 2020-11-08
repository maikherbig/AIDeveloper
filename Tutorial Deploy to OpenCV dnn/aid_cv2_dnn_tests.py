import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  dclab
import cv2
from scipy import ndimage
from keras.models import load_model
import aid_cv2_dnn

#this script contains all functions from AIDeveloper, that are required
#to preprocess images before forwarding through a neural net
def check_10_images(images,pos_x,pos_y,target_imsize):
    
    #preprocess the images
    images = aid_cv2_dnn.image_preprocessing(images,pos_x=pos_x,pos_y=pos_y,pix=1,
                                 target_imsize=target_imsize,
                                 target_channels=1,
                                 zoom_factor=1.0,
                                 zoom_interpol_method=None,
                                 padding_mode="cv2.BORDER_CONSTANT",
                                 normalization_method="Div. by 255",
                                 mean_trainingdata=None,
                                 std_trainingdata=None)
    
    #Check that images have the correct dimension
    assert images.shape[1]==target_imsize
    assert images.shape[2]==target_imsize

    #The best check is to have a look
    fig = plt.figure(figsize=(10, 1))
    columns = 10
    rows = 1
    indx = 0
    for i in range(1, columns*rows +1):
        img = images[indx,:,:,0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img,cmap="gray")
        plt.axis('off')
        indx+=1
    plt.show()


def test_image_preprocessing():
    """
    Create grayscale (A) or RGB (B) images which reflect all possible phenotypes:
        
    1) raw image has odd width, but target image should have even width
    2) raw image has odd height, but target image should have even height
    3) raw image has odd width, and target image should also have odd width
    4) raw image has odd height, and target image should also have odd height

    for each of 1,2,3,4, test following conditions:
        
    a) cell too far on the left
    b) cell too far on the right
    c) cell too far on top
    d) cell too far on bottom
    e) cell placed in each corner (4 cases)
    
    f) target image wider than orignal image
    g) target image higher than orignal image
    h) target image wider and higher than orignal image
        
    """
    
    #load an array that shows a smiley
    smile = np.load("sunglasses_32pix.npy") #save as numpy array

    #A: Grayscale
    smile = (0.21 * smile[:,:,:1]) + (0.72 * smile[:,:,1:2]) + (0.07 * smile[:,:,-1:])
    smile = smile[:,:,0] 
    smile = smile.astype(np.uint8)           

    def test_A(h,w,target_imsize):

        ###A1a (cells placed very far left)
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position very far left:
        pos_x = np.random.randint(low=16,high=24,size=images.shape[0])
        #arbitrary height
        pos_y = np.random.randint(low=16,high=h-16,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
        
        ###A1b (cells placed very far right)
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position very far right:
        pos_x = np.random.randint(low=w-24,high=w-16,size=images.shape[0])
        #arbitrary height
        pos_y = np.random.randint(low=16,high=h-16,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
    
        ###A1c (cells placed on top)
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position random:
        pos_x = np.random.randint(low=16,high=w-16,size=images.shape[0])
        #cells very far up:
        pos_y = np.random.randint(low=16,high=20,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
    
        ###A1d (cells placed on bottom)
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position random:
        pos_x = np.random.randint(low=16,high=w-16,size=images.shape[0])
        #cells very far down:
        pos_y = np.random.randint(low=h-24,high=h-16,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
    
        
        ###A1e (cells in each corner)
        ###Left  upper corner
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position very far left:
        pos_x = np.random.randint(low=16,high=24,size=images.shape[0])
        #quite high
        pos_y = np.random.randint(low=16,high=20,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
        
        ###Right upper corner
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position very far right:
        pos_x = np.random.randint(low=w-24,high=w-16,size=images.shape[0])
        #cells very far up:
        pos_y = np.random.randint(low=16,high=20,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
    
        ###Right lower corner    
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position left:
        pos_x = np.random.randint(low=w-24,high=w-16,size=images.shape[0])
        #cells very far down:
        pos_y = np.random.randint(low=h-24,high=h-16,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)
    
        ###Left lower corner    
        #random noise
        images = np.random.randint(low=0,high=255,size=(10,h,w)).astype(np.uint8)
        #x position left:
        pos_x = np.random.randint(low=16,high=24,size=images.shape[0])
        #cells very far down:
        pos_y = np.random.randint(low=h-24,high=h-16,size=images.shape[0])
        #put smileys on images
        for i in range(len(images)):
            images[i][pos_y[i]-16:pos_y[i]+16,pos_x[i]-16:pos_x[i]+16] = smile
        check_10_images(images,pos_x,pos_y,target_imsize)


    #1) raw image has odd width, but target image should have even width
    test_A(h=80,w=249,target_imsize=48)
    
    #2) raw image has odd height, but target image should have even height
    test_A(h=79,w=250,target_imsize=48)

    #3) raw image has odd width, and target image should also have odd width
    test_A(h=80,w=249,target_imsize=47)

    #4) raw image has odd height, and target image should also have odd height
    test_A(h=79,w=250,target_imsize=47)

    #f) target image wider than orignal image
    test_A(h=79,w=48,target_imsize=64)
    
    #g) target image higher than orignal image
    test_A(h=48,w=249,target_imsize=64)

    #h) target image wider and higher than orignal image
    test_A(h=57,w=73,target_imsize=129)

test_image_preprocessing()

def test_opencv_dnn(rtdc_path,model_keras_path,model_pb_path,meta_path):
    """
    Test if original model and frozen model (.pb) return the same predictions
    Keras is used for the original model (.model) and OpenCV is used for the frozen model (.pb)

    Parameters
    ----------    
    meta_path: str; path to a meta file (generated by AID when fitting a model)
    rtdc_path: str; path to an rtdc file (to get an example image)
    model_pb_path: str; path to a frozen model (.pb) which was generated using AID's model conversion tool
    model_keras_path: str; path to a .model file. This is the default model format of AID and is always generated when fitting a model
    """
    #Load the meta-file to get information about the model
    meta = pd.read_excel(meta_path,sheet_name="Parameters")
    
    try:
        target_imsize = meta["Input image crop"].iloc[0]#input dimensions of the model
    except:
        target_imsize = meta["Input image size"].iloc[0]#input dimensions of the model

    normalization_method = meta["Normalization"].iloc[0]#normalization method
    if normalization_method == "StdScaling using mean and std of all training data":                                
        mean_trainingdata = meta["Mean of training data used for scaling"]
        std_trainingdata = meta["Std of training data used for scaling"]
    else:
        mean_trainingdata = None
        std_trainingdata = None
    
    zoom_factor = 1#should images be zoomed before forwarding through neural net?
    
    #Following parameters may not exist in meta files of older AID versions. Hence try/except
    try:#Define the order for the zooming operation
        zoom_interpol_method = meta["Zoom order"].iloc[0]
    except:
        zoom_interpol_method = "cv2.INTER_NEAREST"
    try:#In case the grabbed images are smaller than model_in_dim, how should the image be padded to increase iamge size 
        padding_mode = meta["paddingMode"].iloc[0]
    except:
        padding_mode = cv2.BORDER_CONSTANT
    try:#Color mode: grayscale or RGB?
        target_channels = meta["Color Mode"].iloc[0]
    except:
        target_channels = "grayscale"
    if target_channels.lower() =="grayscale":
        target_channels = 1
    elif target_channels.lower() =="rgb":
        target_channels = 3

    #Grab the images
    rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
    
    pix = rtdc_ds.config["imaging"]["pixel size"] #get pixelation (um/pix)
    images = rtdc_ds["image"] #get the images
    images = np.array(images)
    pos_x,pos_y = rtdc_ds["pos_x"][:],rtdc_ds["pos_y"][:] 

    images = image_preprocessing(images,pos_x=pos_x,pos_y=pos_y,pix=pix,
                                 target_imsize=target_imsize,
                                 target_channels=target_channels,
                                 zoom_factor=zoom_factor,
                                 zoom_interpol_method=zoom_interpol_method,
                                 padding_mode=padding_mode,
                                 normalization_method=normalization_method,
                                 mean_trainingdata=mean_trainingdata,
                                 std_trainingdata=std_trainingdata)


    #Load the model
    model_pb = cv2.dnn.readNet(model_pb_path)
    blob = cv2.dnn.blobFromImages(images, 1, (target_imsize,target_imsize), swapRB=False, crop=False)
    model_pb.setInput(blob)
    output_pb = model_pb.forward()
    #print("output_pb:"+str(output_pb))
    
    
    #load same model using keras
    model_keras = load_model(model_keras_path)
    output_keras = model_keras.predict(images)
    #print("output_keras:"+str(output_keras))
    
    assert np.allclose(output_pb,output_keras)


def pad_functions_compare(arguments):
    """
    numpy's pad and OpenCVs copyMakeBorder can do the same thing, but the 
    function arguments are called differntly.

    Find out, wich sets of arguments lead to the same result
    
    """
    images = np.random.randint(low=0,high=255,size=(80,250)).astype(np.uint8)
    
    #common arguments:
    top,bottom,left,right = 4,5,6,7   
    
    #np.pad argument
    padding_mode = 'maximum' 
    
    #cv2.copyMakeBorder arguments
    borderType = cv2.BORDER_CONSTANT

    img_pad_np = np.pad(images,pad_width=( (top, bottom),(left, right) ),mode=padding_mode)
    
    value = [255,110]#np.arange(0,images.shape[0])
    img_pad_cv2 = cv2.copyMakeBorder(images, top, bottom, left, right, borderType,value=value)

    compare = img_pad_np==img_pad_cv2
    assert compare.all(), "images returned from np.pad and cv2.copyMakeBorder are not identical!"

def pad_arguments_np2cv(padding_mode):
    """
    numpy's pad and OpenCVs copyMakeBorder can do the same thing, but the 
    function arguments are called differntly.

    This function takes numpy padding_mode argument and returns the 
    corresponsing borderType for cv2.copyMakeBorder

    Parameters
    ---------- 
    padding_mode: str; numpy padding mode
        - "constant" (default): Pads with a constant value.
        - "edge": Pads with the edge values of array.
        - "linear_ramp": Pads with the linear ramp between end_value and the array edge value.
        - "maximum": Pads with the maximum value of all or part of the vector along each axis.
        - "mean": Pads with the mean value of all or part of the vector along each axis.
        - "median": Pads with the median value of all or part of the vector along each axis.
        - "minimum": Pads with the minimum value of all or part of the vector along each axis.
        - "reflect": Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
        - "symmetric": Pads with the reflection of the vector mirrored along the edge of the array.
        - "wrap": Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

    Returns
    ----------   
    str: OpenCV borderType     
        - "cv2.BORDER_CONSTANT": iiiiii|abcdefgh|iiiiiii with some specified i 
        - "cv2.BORDER_REFLECT": fedcba|abcdefgh|hgfedcb
        - "cv2.BORDER_REFLECT_101": gfedcb|abcdefgh|gfedcba
        - "cv2.BORDER_DEFAULT": same as BORDER_REFLECT_101
        - "cv2.BORDER_REPLICATE": aaaaaa|abcdefgh|hhhhhhh
        - "cv2.BORDER_WRAP": cdefgh|abcdefgh|abcdefg
    """

    #Check that the padding_mode is actually supported by OpenCV
    supported = ["constant","edge","reflect","symmetric","wrap"]
    assert padding_mode in supported, "The padding mode: '"+padding_mode+"' is\
 not supported"
    
    if padding_mode=="constant":
        return "cv2.BORDER_CONSTANT"
    if padding_mode=="edge":
        return "cv2.BORDER_REPLICATE"
    if padding_mode=="reflect":
        return "cv2.BORDER_REFLECT_101"
    if padding_mode=="symmetric":
        return "cv2.BORDER_REFLECT"
    if padding_mode=="wrap":
        return "cv2.BORDER_WRAP"
    
def zoom_functions_compare():
    """
    Scipy and OpenCV provide functions to zoom images
    Find out, which arguments lead to identical results.

    zoom_interpol_method: an OpenCV interpolation flag
        can be one of the following methods:
        - cv2.INTER_NEAREST: a nearest-neighbor interpolation 
        - cv2.INTER_LINEAR: a bilinear interpolation (used by default) 
        - cv2.INTER_AREA: resampling using pixel area relation. It may be a 
        preferred method for image decimation, as it gives moire’-free results. 
        But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        - cv2.INTER_CUBIC: a bicubic interpolation over 4×4 pixel neighborhood 
        - cv2.INTER_LANCZOS4: a Lanczos interpolation over 8×8 pixel neighborhood        
 
    """


    """ 
    I could'nt find any set of function arguments where scipy and OpenCV returned 
    identical images! Hence, I conducted a pixel-wise comparison to find methods,
    that return at least somewhat similar images
    """

    images = np.random.randint(low=0,high=255,size=(80,250)).astype(np.uint8)
    #SciPy settings
    zoom_factor = 20
    zoom_order = 1
    #OpenCV settings
    #final_h = int(np.around(images.shape[0]*zoom_factor))
    #final_w = int(np.around(images.shape[1]*zoom_factor))
    #interpolation_method = cv2.INTER_NEAREST
    
    #Perform the zooming operations
    for zoom_order in [0,1,2,3,4,5]:
        zoomed_scipy = ndimage.zoom(images, zoom=(zoom_factor,zoom_factor),order=int(zoom_order))
        operations = ["cv2.INTER_NEAREST","cv2.INTER_LINEAR","cv2.INTER_AREA","cv2.INTER_CUBIC","cv2.INTER_LANCZOS4"]
        Best = []
        for operation in operations:            
            #print(operation)
            #zoomed_cv2 = cv2.resize(images, dsize=(final_w,final_h), interpolation=interpolation_method)
            zoomed_cv2 = cv2.resize(images, dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=eval(operation))
    
            diff = abs(zoomed_scipy-zoomed_cv2)
            Best.append(np.sum(diff))
            #print(np.sum(diff))
            #plt.imshow(diff)
            #plt.show()
            
            #compare = zoomed_scipy==zoomed_cv2
            #assert compare.all(), "images returned from ndimage.zoom and cv2.resize are not identical!"
    
        ind = np.argmin(np.array(Best))
        print("For zoom_order="+str(zoom_order)+", "+str(operations[ind])+" is closest")
    
    """
    Results:
    for zoom_factor = 0.1
        For zoom_order=0, cv2.INTER_LINEAR is closest
        For zoom_order=1, cv2.INTER_CUBIC is closest
        For zoom_order=2, cv2.INTER_LANCZOS4 is closest
        For zoom_order=3, cv2.INTER_CUBIC is closest
        For zoom_order=4, cv2.INTER_CUBIC is closest
        For zoom_order=5, cv2.INTER_CUBIC is closest
    for zoom_factor = 0.2
        For zoom_order=0, cv2.INTER_LINEAR is closest
        For zoom_order=1, cv2.INTER_LINEAR is closest
        For zoom_order=2, cv2.INTER_AREA is closest
        For zoom_order=3, cv2.INTER_LINEAR is closest
        For zoom_order=4, cv2.INTER_LINEAR is closest
        For zoom_order=5, cv2.INTER_LINEAR is closest
    for zoom_factor = 0.4
        For zoom_order=0, cv2.INTER_NEAREST is closest
        For zoom_order=1, cv2.INTER_AREA is closest
        For zoom_order=2, cv2.INTER_LINEAR is closest
        For zoom_order=3, cv2.INTER_LINEAR is closest
        For zoom_order=4, cv2.INTER_AREA is closest
        For zoom_order=5, cv2.INTER_AREA is closest   
    for zoom_factor = 0.6
        For zoom_order=0, cv2.INTER_LINEAR is closest
        For zoom_order=1, cv2.INTER_LINEAR is closest
        For zoom_order=2, cv2.INTER_CUBIC is closest
        For zoom_order=3, cv2.INTER_LANCZOS4 is closest
        For zoom_order=4, cv2.INTER_LANCZOS4 is closest
        For zoom_order=5, cv2.INTER_LANCZOS4 is closest
    for zoom_factor = 0.7
        For zoom_order=0, cv2.INTER_NEAREST is closest
        For zoom_order=1, cv2.INTER_LINEAR is closest
        For zoom_order=2, cv2.INTER_LANCZOS4 is closest
        For zoom_order=3, cv2.INTER_LANCZOS4 is closest
        For zoom_order=4, cv2.INTER_LANCZOS4 is closest
        For zoom_order=5, cv2.INTER_LANCZOS4 is closest
    for zoom_factor (s) = 0.8, 1.2, 2.0, 3.0, 6.0, 20.0 same result:
        For zoom_order=0, cv2.INTER_NEAREST is closest
        For zoom_order=1, cv2.INTER_LINEAR is closest
        For zoom_order=2, cv2.INTER_CUBIC is closest
        For zoom_order=3, cv2.INTER_LANCZOS4 is closest
        For zoom_order=4, cv2.INTER_LANCZOS4 is closest
        For zoom_order=5, cv2.INTER_LANCZOS4 is closest
    """
    
def zoom_arguments_scipy2cv(zoom_factor,zoom_interpol_method):
    #Resulting images after performing ndimage.zoom or cv2.resize are never identical
    #But with certain settings you get at least similar results, 

    if zoom_factor>=0.8:
        if zoom_interpol_method==0: return "cv2.INTER_NEAREST"
        elif zoom_interpol_method==1: return "cv2.INTER_LINEAR"
        elif zoom_interpol_method==2: return "cv2.INTER_CUBIC"
        elif zoom_interpol_method==3: return "cv2.INTER_LANCZOS4"
        elif zoom_interpol_method==4: return "cv2.INTER_LANCZOS4"
        elif zoom_interpol_method==5: return "cv2.INTER_LANCZOS4"

    if zoom_factor<=0.8: #for downsampling the image, all methods perform similar
        #but cv2.INTER_LINEAR, is closest most of the time, irrespective of the zoom_order
        return "cv2.INTER_LINEAR"


def comp_time_zoom():
    """
    Compare the computational time of some image zooming operations
    """   
    images = np.random.randint(low=0,high=255,size=(10000,80,250)).astype(np.uint8)
    channels = 1
    zoom_factor = 1.2
    zoom_order = 1
    interpolation_method = cv2.INTER_LINEAR
        
    print("Batch-processing "+str(images.shape[0])+" images:")
    
    t1 = time.time()
    if channels==1:                
        images_zoomed = ndimage.zoom(images, zoom=(1,zoom_factor,zoom_factor),order=int(zoom_order))
    elif channels==3:                
        images_zoomed = ndimage.zoom(images, zoom=(1,zoom_factor,zoom_factor,1),order=int(zoom_order))
    t2 = time.time()
    dt = np.round(t2-t1,1)
    print("scipy ndimage.zoom:" +str(dt))
    
    t1 = time.time()
    final_h = int(np.around(images.shape[1]*zoom_factor))
    final_w = int(np.around(images.shape[2]*zoom_factor))
    images_zoomed = list(images)
    for i in range(len(images)):
        #the order width,height in cv2.resize is not an error. OpenCV wants this order...
        images_zoomed[i] = cv2.resize(images_zoomed[i], dsize=(final_w,final_h), interpolation=interpolation_method)
    t2 = time.time()
    dt = np.round(t2-t1,1)
    print("OpenCV resize:" +str(dt))

    """ 
    This functions returns on my PC (Intel Core i7-4810MO@2.8GHz, 24GB RAM):
    '
    scipy ndimage.zoom:13.5
    OpenCV resize:0.5
    '
    -> despite having to loop through a list, OpenCV is much faster compared to scipy,
    which gets all images at once and could do batch processing!
   """ 

def comp_time_padding():
    """
    Compare the computational time of some padding operations
    """   
    images = np.random.randint(low=0,high=255,size=(10000,80,250)).astype(np.uint8)
    
    padding_mode = 'constant'
    pad = 10
    top,bottom,left,right = pad,pad,pad,pad
    borderType = cv2.BORDER_CONSTANT


    t1 = time.time()
    temp = np.pad(images,pad_width=( (0,0),(pad, pad),(pad, pad) ),mode=padding_mode)
    t2 = time.time()
    dt = np.round(t2-t1,2)
    print("Numpy pad (stack of images): " +str(dt))

    t1 = time.time()
    images = list(images)
    for i in range(len(images)):
        images[i] = np.pad(images[i],pad_width=( (pad, pad),(pad, pad) ),mode=padding_mode)
    t2 = time.time()
    dt = np.round(t2-t1,2)
    print("Numpy pad (loop over images): " +str(dt))
       
    images = np.random.randint(low=0,high=255,size=(10000,80,250)).astype(np.uint8)
    images = list(images)

    t1 = time.time()
    for i in range(len(images)):
        images[i] = cv2.copyMakeBorder(images[i], top, bottom, left, right, borderType)
    t2 = time.time()
    dt = np.round(t2-t1,2)
    print("OpenCV pad (loop over images): " +str(dt))

    """ 
    This functions returns on my PC (Intel Core i7-4810MO@2.8GHz, 24GB RAM):
    '
    Numpy pad (stack of images): 1.01
    Numpy pad (loop over images): 0.86
    OpenCV pad (loop over images): 0.23
    '
    -> Stack processing images in numpy does not make it faster
    -> Using OpenCV and looping through images is fastest,
    
   """ 






def smiley_save_to_np():
    from keras.preprocessing.image import load_img
    #load a smiley png
    smile = load_img("sunglasses_32pix.png")
    smile = np.array(smile) #convert to numpy
    np.save("sunglasses_32pix.npy",smile) #save as numpy array


#Eosinophil dataset
#rtdc_path = "Eos - M1.rtdc"
#Smiley-Blink dataset (10 images)
rtdc_path = r"Smileys\Data\blink_10_gray.rtdc"

#rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
#pix = rtdc_ds.config["imaging"]["pixel size"] #get pixelation (um/pix)
#images = rtdc_ds["image"] #get the images
#images = np.array(images)

target_imsize=24
target_channels=1
zoom_factor=1.0
padding_mode = "symmetric"
padding_mode = "cv2.BORDER_CONSTANT"
zoom_factor = 1.2
zoom_order = 1
zoom_interpol_method = "cv2.INTER_LINEAR"

#Blood model
meta_path = "M09_LeNet5_Mean_Blood_36pix_40x_meta.xlsx"#Path to the meta file which was recorded when the model was trained
model_keras_path = r"M09_LeNet5_Mean_Blood_36pix_40x_1650.model"#Path the the original model (keras hdf5 format)
model_pb_path = 'M09_LeNet5_Mean_Blood_36pix_40x_1650_optimized.pb'#Path to the frozen model

preds_cnn_blood = test_forward_dnn(rtdc_path,model_pb_path,meta_path)

#Smiley model MLP64 grayscale
meta_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_gray_meta.xlsx"#Path to the meta file which was recorded when the model was trained
model_keras_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_gray_9479.model"#Path the the original model (keras hdf5 format)
model_pb_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_gray_9479_optimized.pb"#Path to the frozen model

preds_mlp_gray = test_forward_dnn(rtdc_path,model_pb_path,meta_path)

#Smiley model MLP64 rgb
meta_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_rgb_meta.xlsx"#Path to the meta file which was recorded when the model was trained
model_keras_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_rgb_9912.model"#Path the the original model (keras hdf5 format)
model_pb_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\MLP64_rgb_9912_optimized.pb"#Path to the frozen model

preds_mlp_rgb = test_forward_dnn(rtdc_path,model_pb_path,meta_path)

#Smiley model MLP64 grayscale
meta_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_gray_meta.xlsx"#Path to the meta file which was recorded when the model was trained
model_keras_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_gray_9259.model"#Path the the original model (keras hdf5 format)
model_pb_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_gray_9259_optimized.pb"#Path to the frozen model

preds_cnn_gray = test_forward_dnn(rtdc_path,model_pb_path,meta_path)

#Smiley model MLP64 rgb
meta_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_rgb_meta.xlsx"#Path to the meta file which was recorded when the model was trained
model_keras_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_rgb_9321.model"#Path the the original model (keras hdf5 format)
model_pb_path = r"D:\BIOTEC-WORK\Paul - ShapeOut\04_AID_models_OpenCV_dnn\02_for_dclab\Smileys\Models_v02\\LeNet_bn_do_rgb_9321_optimized.pb"#Path to the frozen model

preds_cnn_rgb = test_forward_dnn(rtdc_path,model_pb_path,meta_path)



#test_opencv_dnn(rtdc_path,model_keras_path,model_pb_path,meta_path)
    

#Load the model to a global variable
#model_pb = cv2.dnn.readNet(model_pb_path)
#Load  image preprocessing settings corresponding to the model to a global var
#img_processing_settings = load_model_meta(meta_path)


rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)

pix = rtdc_ds.config["imaging"]["pixel size"] #get pixelation (um/pix)
images = rtdc_ds["image"] #get the images
images = np.array(images)
pos_x,pos_y = rtdc_ds["pos_x"][:],rtdc_ds["pos_y"][:] 


import os,time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  dclab
import cv2
from scipy import ndimage
from keras.models import load_model

#this script contains all functions from AIDeveloper, that are required
#to preprocess images before forwarding through a neural net

def image_adjust_channels(images,target_channels=1):
    """
    Check the number of channels of images.
    Transform images (if needed) to get to the desired number of channels
    
    Parameters
    ----------
    images: numpy array of dimension (nr.images,height,width) for grayscale,
    or of dimension (nr.images,height,width,channels) for RGB images

    target_channels: int
        target number of channels
        can be one of the following:
        
        - 1: target is a grayscale image. In case RGB images are 
        provided, the luminosity formula is used to convert of RGB to 
        grayscale
        - 3: target is an RGB image. In case grayscale images are provided,
        the information of each image is copied to all three channels to 
        convert grayscale to RGB"
    
    Returns
    ----------
    images: numpy array
        images with adjusted number of channels
    """

    #images.shape is (N,H,W) for grayscale, or (N,H,W,C) for RGB images
    #(N,H,W,C) means (nr.images,height,width,channels)

    #Make sure either (N,H,W), or (N,H,W,C) is provided
    assert len(images.shape)==4 or len(images.shape)==3, "Shape of 'images' \
    is not supported: " +str(images.shape) 

    if len(images.shape)==4:#Provided images are RGB
        #Mare sure there are 3 channels (RGB)
        assert images.shape[-1]==3, "Images have "+str(images.shape[-1])+" channels. This is (currently) not supported!"

        if target_channels==1:#User wants Grayscale -> use the luminosity formula
            images = (0.21 * images[:,:,:,:1]) + (0.72 * images[:,:,:,1:2]) + (0.07 * images[:,:,:,-1:])
            images = images[:,:,:,0] 
            images = images.astype(np.uint8)           
            print("Used luminosity formula to convert RGB to Grayscale")
            
    if len(images.shape)==3:#Provided images are Grayscale
        if target_channels==3:#User wants RGB -> copy the information to all 3 channels
            images = np.stack((images,)*3, axis=-1)
            print("Copied information to all three channels to convert Grayscale to RGB")
    return images


def image_crop_pad_np(images,pos_x,pos_y,final_h,final_w,padding_mode='constant'):
    """
    Deprecated: Please use 'image_crop_pad_cv' instead, which uses OpenCV instead of numpy.
    
    Function takes a list images (list of numpy arrays) an resizes them to 
    equal size by center cropping and/or padding.

    Parameters
    ----------
    images: list of images of arbitrary shape
    (nr.images,height,width,channels) 
        can be a single image or multiple images
    final_h: int
        target image height [pixels]
        
    final_w: int
        target image width [pixels]
        
    padding_mode: str
        Perform the following padding operation if the cell is too far at the 
        border of the image such that the  desired image size cannot be 
        obtained without going beyond the order of the image:
        
        - Delete: Return empty array (all zero) if the cell is too far at border (delete image)
        
        #the following text is copied from 
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html</p></body></html>"
        
        - constant (default): Pads with a constant value.
        - edge: Pads with the edge values of array.
        - linear_ramp: Pads with the linear ramp between end_value and the array edge value.
        - maximum: Pads with the maximum value of all or part of the vector along each axis.
        - mean: Pads with the mean value of all or part of the vector along each axis.
        - median: Pads with the median value of all or part of the vector along each axis.
        - minimum: Pads with the minimum value of all or part of the vector along each axis.
        - reflect: Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
        - symmetric: Pads with the reflection of the vector mirrored along the edge of the array.
        - wrap: Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

    
    Returns
    ----------
    images: list of images. Each image is a numpy array of shape 
    (final_h,final_w,channels) 
    """
    print("Deprecated: Please use 'image_crop_pad_cv' instead, which uses OpenCV instead of numpy.")
    
    for i in range(len(images)):
        image = images[i]
    
        #Compute the edge-coordinates that define the cropped image
        y1 = np.around(pos_y[i]-final_h/2.0)              
        x1 = np.around(pos_x[i]-final_w/2.0) 
        y2 = y1+final_h               
        x2 = x1+final_w

        #Are these coordinates within the oringinal image?
        #If not, the image needs padding
        pad_top,pad_bottom,pad_left,pad_right = 0,0,0,0

        if y1<0:#Padding is required on top of image
            pad_top = int(abs(y1))
            y1 = 0 #set y1 to zero and pad pixels after cropping
            
        if y2>image.shape[0]:#Padding is required on bottom of image
            pad_bottom = int(y2-image.shape[0])
            y2 = image.shape[0]
        
        if x1<0:#Padding is required on left of image
            pad_left = int(abs(x1))
            x1 = 0
        
        if x2>image.shape[1]:#Padding is required on right of image
            pad_right = int(x2-image.shape[1])
            x2 = image.shape[1]
        
        #Get cropped image
        temp = image[int(y1):int(y2),int(x1):int(x2)]

        if pad_top+pad_bottom+pad_left+pad_right>0:
            if padding_mode=="Delete":
                temp = np.zeros_like(temp)
            else:
                #Perform all padding operations in one go
                temp = np.pad(temp,pad_width=( (pad_top, pad_bottom),(pad_left, pad_right) ),mode=padding_mode)

        images[i] = temp
            
    return images

def image_crop_pad_cv2(images,pos_x_pix,pos_y_pix,final_h,final_w,padding_mode="cv2.BORDER_CONSTANT"):
    """
    Function takes a list images (list of numpy arrays) an resizes them to 
    equal size by center cropping and/or padding.

    Parameters
    ----------
    images: list of images of arbitrary shape
    (nr.images,height,width,channels) 
        can be a single image or multiple images
    pos_x_pix: float or ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [pixels]
    pos_y_pix: float or ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [pixels]
        
    final_h: int
        target image height [pixels]
    
    final_w: int
        target image width [pixels]
        
    padding_mode: str; OpenCV BorderType
        Perform the following padding operation if the cell is too far at the 
        border of the image such that the  desired image size cannot be 
        obtained without going beyond the order of the image:
        
        - "Delete": Return empty array (all zero) if the cell is too far at border (delete image)
        
        #the following text is copied from 
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
        
        - "cv2.BORDER_CONSTANT": iiiiii|abcdefgh|iiiiiii with some specified i 
        - "cv2.BORDER_REFLECT": fedcba|abcdefgh|hgfedcb
        - "cv2.BORDER_REFLECT_101": gfedcb|abcdefgh|gfedcba
        - "cv2.BORDER_DEFAULT": same as BORDER_REFLECT_101
        - "cv2.BORDER_REPLICATE": aaaaaa|abcdefgh|hhhhhhh
        - "cv2.BORDER_WRAP": cdefgh|abcdefgh|abcdefg
    
    Returns
    ----------
    images: list of images. Each image is a numpy array of shape 
    (final_h,final_w,channels) 

    """
    for i in range(len(images)):
        image = images[i]
    
        #Compute the edge-coordinates that define the cropped image
        y1 = np.around(pos_y_pix[i]-final_h/2.0)              
        x1 = np.around(pos_x_pix[i]-final_w/2.0) 
        y2 = y1+final_h               
        x2 = x1+final_w

        #Are these coordinates within the oringinal image?
        #If not, the image needs padding
        pad_top,pad_bottom,pad_left,pad_right = 0,0,0,0

        if y1<0:#Padding is required on top of image
            pad_top = int(abs(y1))
            y1 = 0 #set y1 to zero and pad pixels after cropping
            
        if y2>image.shape[0]:#Padding is required on bottom of image
            pad_bottom = int(y2-image.shape[0])
            y2 = image.shape[0]
        
        if x1<0:#Padding is required on left of image
            pad_left = int(abs(x1))
            x1 = 0
        
        if x2>image.shape[1]:#Padding is required on right of image
            pad_right = int(x2-image.shape[1])
            x2 = image.shape[1]
        
        #Crop the image
        temp = image[int(y1):int(y2),int(x1):int(x2)]

        if pad_top+pad_bottom+pad_left+pad_right>0:
            if padding_mode=="Delete":
                temp = np.zeros_like(temp)
            else:
                #Perform all padding operations in one go
                temp = cv2.copyMakeBorder(temp, pad_top, pad_bottom, pad_left, pad_right, eval(padding_mode))
        
        images[i] = temp
            
    return images


def image_zooming(images,zoom_factor,zoom_interpol_method):
    """
    Function takes a list of images (list of numpy arrays) an resizes them to 
    an equal size by scaling (interpolation).
    
    images: list of images of arbitrary shape
    zoom_factor: float
        Factor by which the size of the images should be zoomed
    zoom_interpol_method: str; available are: (text copied from original docs: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize)
        -"cv2.INTER_NEAREST" – a nearest-neighbor interpolation
        -"cv2.INTER_LINEAR" – a bilinear interpolation (used by default)
        -"cv2.INTER_AREA" – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        -"cv2.INTER_CUBIC" - a bicubic interpolation over 4x4 pixel neighborhood
        -"cv2.INTER_LANCZOS4" - a Lanczos interpolation over 8x8 pixel neighborhood
    """

    final_h = int(np.around(zoom_factor*images[0].shape[1]))
    final_w = int(np.around(zoom_factor*images[0].shape[2]))

    for i in range(len(images)):
        #the order (width,height) in cv2.resize is not an error. OpenCV wants this order...
        images[i] = cv2.resize(images[i], dsize=(final_w,final_h), interpolation=eval(zoom_interpol_method))
    return images

def image_normalization(images,normalization_method,mean_trainingdata=None,std_trainingdata=None):
    if normalization_method == "StdScaling using mean and std of all training data":
        #make sure pandas series is converted to numpy array
        if type(mean_trainingdata)==pd.core.series.Series:
            mean_trainingdata=mean_trainingdata.values[0]
            std_trainingdata=std_trainingdata.values[0]
        if np.allclose(std_trainingdata,0):
            std_trainingdata = 0.0001
            print("Set the standard deviation (std_trainingdata) to 0.0001 because otherwise div. by 0 would have happend!")

    if len(images.shape)==3: #single channel Grayscale rtdc data
        #Add the "channels" dimension
        images = np.expand_dims(images,3)    
    images = images.astype(np.float32)
    
    for k in range(images.shape[0]):
        line = images[k,:,:,:]
        ###########Scaling############
        if normalization_method == "None":
            pass #dont do anything
        elif normalization_method == "Div. by 255":
            line = line/255.0
        elif normalization_method == "StdScaling using mean and std of each image individually":
            mean = np.mean(line)
            std = np.std(line)
            if np.allclose(std,0):
                std = 0.0001
                print("Set the standard deviation to 0.0001 because otherwise div. by 0 would have happend!")
            line = (line-mean)/std
        elif normalization_method == "StdScaling using mean and std of all training data":
            line = (line-mean_trainingdata)/std_trainingdata
            
        #Under NO circumstances, training data should contain nan values
        ind = np.isnan(line)
        line[ind] = np.random.random() #replace nan with random values. This is better than nan, since .fit will collapse and never get back
        images[k,:,:,:] = line   
    return images        


def image_preprocessing(images,pos_x,pos_y,pix=0.34,target_imsize=32,
                        target_channels=1,zoom_factor=1,
                        zoom_interpol_method="cv2.INTER_NEAREST",
                        padding_mode="cv2.BORDER_CONSTANT",
                        normalization_method="Div. by 255",
                        mean_trainingdata=None,std_trainingdata=None):
    """
    Wrapper function which performs all image preprocessing steps, required
    for preparing raw rtdc images to be forwarded through a neural net,
    
    Parameters
    ----------
    images: numpy array of shape (nr.images,height,width,channels) 
        can be a single image or multiple images
    pos_x: float or ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [um]
    pos_y: float or ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [um]
    pix: float
        Resolution [µm/pix]
    target_imsize: int
        target image size (in pixels)
        currently, only squared images are supported. Hence, width=height
    target_channels: int
        Indicates the number of channels of the images
        can be one of the following:
        - 1: model expects grayscale images 
        - 3: model expects RGB (color) images 
    zoom_factor: float
        Factor by which the size of the images should be zoomed
    zoom_interpol_method: str; OpenCV interpolation flag
        can be one of the following methods:
        - "cv2.INTER_NEAREST": a nearest-neighbor interpolation 
        - "cv2.INTER_LINEAR": a bilinear interpolation (used by default) 
        - "cv2.INTER_AREA": resampling using pixel area relation. It may be a 
        preferred method for image decimation, as it gives moire’-free results. 
        But when the image is zoomed, it is similar to the INTER_NEAREST method. 
        - "cv2.INTER_CUBIC": a bicubic interpolation over 4×4 pixel neighborhood 
        - "cv2.INTER_LANCZOS4": a Lanczos interpolation over 8×8 pixel neighborhood        

    padding_mode: str; OpenCV BorderType
        Perform the following padding operation if the cell is too far at the 
        border of the image such that the  desired image size cannot be 
        obtained without going beyond the order of the image:
        
        - "Delete": Return empty array (all zero) if the cell is too far at border (delete image)
        
        #the following text is copied from 
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
        
        - "cv2.BORDER_CONSTANT": iiiiii|abcdefgh|iiiiiii with some specified i 
        - "cv2.BORDER_REFLECT": fedcba|abcdefgh|hgfedcb
        - "cv2.BORDER_REFLECT_101": gfedcb|abcdefgh|gfedcba
        - "cv2.BORDER_DEFAULT": same as BORDER_REFLECT_101
        - "cv2.BORDER_REPLICATE": aaaaaa|abcdefgh|hhhhhhh
        - "cv2.BORDER_WRAP": cdefgh|abcdefgh|abcdefg
    
    normalization_method: str
        Define a method to normalize the pixel-values of the images.
        can be one of the following methods:
        - 'None': No normalization is applied
        - 'Div. by 255': each pixel value is divided by 255 (default)
        - 'StdScaling using mean and std of each image individually': The mean and 
        standard deviation of each input image itself is used to scale it by 
        first subtracting the mean and then dividing by the standard deviation.
        -StdScaling using mean and std of all training data: A mean and std. 
        value was obtained while fitting the neural net by averaging the entire 
        training dataset. These fixed values are used to scale images during 
        training by first subtracting the mean and then dividing by the 
        standard deviation.
    """

    #Adjust number of channels
    images = image_adjust_channels(images,target_channels)
    #Convert image array to list 
    images = list(images)

    #Convert position of cell from "um" to "pixel index"
    pos_x,pos_y = pos_x/pix,pos_y/pix  

    #Apply zooming operation if required
    if zoom_factor!=1:
        images = image_zooming(images,zoom_factor,zoom_interpol_method)
        #Adjust pos_x and pos_y accordingly
        pos_x,pos_y = zoom_factor*pos_x,zoom_factor*pos_y

    #Cropping and padding operation to obtain images of desired size
    images = image_crop_pad_cv2(images=images,pos_x_pix=pos_x,pos_y_pix=pos_y,final_h=target_imsize,final_w=target_imsize,padding_mode=padding_mode)

    #Convert to unit8 arrays
    images = np.array((images), dtype="uint8")

    if target_channels==1:#User wants Grayscale -> add empty channel dimension (required for inference in OpenCV dnn)
        images = np.expand_dims(images,3)

    #Normalize images
    images = image_normalization(images,normalization_method="Div. by 255",mean_trainingdata=None,std_trainingdata=None)

    return images

def load_model_meta(meta_path):
    """
    Extract meta information from a meta file that was created during training 
    in AID. Function returns all information how images need ot be preprocessed 
    before passing them throught the neural net. 

    Parameters
    ----------    
    meta_path: str; path to a meta file (generated by AID when during model training)

    Returns
    ----------    
    pd.DataFrame ; A DataFrame with the following keys:
        target_imsize: input image size required by the neural net
        target_channels: number of image channels required by the neural net
        normalization_method: the method to normalize the pixel-values of the images
        mean_trainingdata: the mean pixel value obtained from the training dataset
        std_trainingdata: the std of the pixel values obtained from the training dataset
        zoom_factor: factor by which the size of the images should be zoomed
        zoom_interpol_method: OpenCV interpolation flag
        padding_mode: OpenCV borderType flag
    """
    xlsx = pd.ExcelFile(meta_path)
    
    #The zooming factor is saved in the UsedData sheet
    meta = pd.read_excel(xlsx,sheet_name="UsedData")
    zoom_factor = meta["zoom_factor"].iloc[0]#should images be zoomed before forwarding through neural net?

    meta = pd.read_excel(xlsx,sheet_name="Parameters")
    
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
  
    #Following parameters may not exist in meta files of older AID versions. Hence try/except

    #Color mode: grayscale or RGB?
    try:
        target_channels = meta["Color Mode"].iloc[0]
    except:
        target_channels = "grayscale"
    if target_channels.lower() =="grayscale":
        target_channels = 1
    elif target_channels.lower() =="rgb":
        target_channels = 3

    #The order for the zooming operation
    try:
        zoom_interpol_method = meta["Zoom order"].iloc[0]
    except:
        zoom_interpol_method = "cv2.INTER_NEAREST"
    #Translate zoom_interpol_method to OpenCV argument
    if "cv2." not in str(zoom_interpol_method):
        zoom_interpol_method = zoom_arguments_scipy2cv(zoom_factor,zoom_interpol_method)

    #Padding mode
    try:
        padding_mode = meta["paddingMode"].iloc[0]
    except:
        padding_mode = "constant"#cv2.BORDER_CONSTANT
    #translate padding_mode to OpenCV argument
    if "cv2." not in padding_mode:
        padding_mode = pad_arguments_np2cv(padding_mode)

    #Write information in one DataFrame
    img_processing_settings = pd.DataFrame()
    img_processing_settings["target_imsize"]=target_imsize,
    img_processing_settings["target_channels"]=target_channels,
    img_processing_settings["normalization_method"]=normalization_method,
    img_processing_settings["mean_trainingdata"]=mean_trainingdata,
    img_processing_settings["std_trainingdata"]=std_trainingdata,
    img_processing_settings["zoom_factor"]=zoom_factor,
    img_processing_settings["zoom_interpol_method"]=zoom_interpol_method,
    img_processing_settings["padding_mode"]=padding_mode,
    
    return img_processing_settings


def forward_images_cv2(model_pb,img_processing_settings,images,pos_x,pos_y,pix):
    
    target_imsize = int(img_processing_settings["target_imsize"].values[0])
    target_channels = int(img_processing_settings["target_channels"].values[0])
    zoom_factor = float(img_processing_settings["zoom_factor"].values[0])
    zoom_interpol_method = str(img_processing_settings["zoom_interpol_method"].values[0])
    padding_mode = str(img_processing_settings["padding_mode"].values[0])
    normalization_method = str(img_processing_settings["normalization_method"].values[0])
    mean_trainingdata = img_processing_settings["mean_trainingdata"].values[0]
    std_trainingdata = img_processing_settings["std_trainingdata"].values[0]

    #Preprocess images
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
    blob = cv2.dnn.blobFromImages(images, 1, (target_imsize,target_imsize), swapRB=False, crop=False)
    model_pb.setInput(blob)
    output_pb = model_pb.forward()
    return output_pb
    
    
    



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











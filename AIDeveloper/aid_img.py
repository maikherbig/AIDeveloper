# -*- coding: utf-8 -*-
"""
aid_img
some functions for image processing that are essential for AIDeveloper
---------
@author: maikherbig
"""

import numpy as np
import os, shutil,h5py
import pandas as pd
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import aid_bin
from pyqtgraph.Qt import QtWidgets
#from scipy import ndimage
import cv2
import tensorflow as tf
from tensorflow.python.client import device_lib
device_types = device_lib.list_local_devices()
device_types = [device_types[i].device_type for i in range(len(device_types))]
config_gpu = tf.compat.v1.ConfigProto()
if device_types[0]=='GPU':
    config_gpu.gpu_options.allow_growth = True
    config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.7

dir_root = os.path.dirname(aid_bin.__file__)#ask the module for its origin

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

def zoom_arguments_scipy2cv(zoom_factor,zoom_interpol_method):
    """
    Resulting images after performing ndimage.zoom or cv2.resize are never identical,
    but with certain settings you get at least similar results. 
    Parameters
    ----------    
    zoom_factor: float, 
        factor by which the size of the images should be zoomed
    zoom_interpol_method: int, 
        The order of the spline interpolation
    
    Returns
    ----------    
    str; OpenCV interpolation flag
    """
    
    #Case 1: AIDeveloper>=0.2.1: zoom_interpol_method is already an OpenCV interpolation flag
    opencv_zoom_options = ["cv2.INTER_NEAREST","cv2.INTER_LINEAR","cv2.INTER_AREA","cv2.INTER_CUBIC","cv2.INTER_LANCZOS4"]
    if type(zoom_interpol_method)==str and "cv2" in zoom_interpol_method: 
        ind = [o in zoom_interpol_method for o in opencv_zoom_options]
        ind = np.where(np.array(ind)==True)[0][0]
        return opencv_zoom_options[ind]
    
    #Case 2: AIDeveloper<0.2.1: zoom_interpol_method is ment for scipy, conversion needed
    if type(zoom_interpol_method)==str and "cv2" not in zoom_interpol_method: 
        if zoom_interpol_method == "0 (nearest neighbor)":
            zoom_interpol_method = 0
        elif zoom_interpol_method == "1 (lin. interp.)":
            zoom_interpol_method = 1
        elif zoom_interpol_method == "2 (quadr. interp.)":
            zoom_interpol_method = 2
        elif zoom_interpol_method == "3 (cubic interp.)":
            zoom_interpol_method = 3
        elif zoom_interpol_method == "4":
            zoom_interpol_method = 4
        elif zoom_interpol_method == "5":
            zoom_interpol_method = 5
    
    #depending on the zoom_factor, a certain OpenCV interpolation flag should be used
    if zoom_factor>=0.8:
        if zoom_interpol_method==0: return "cv2.INTER_NEAREST"
        elif zoom_interpol_method==1: return "cv2.INTER_LINEAR"
        elif zoom_interpol_method==2: return "cv2.INTER_CUBIC"
        elif zoom_interpol_method==3: return "cv2.INTER_LANCZOS4"
        elif zoom_interpol_method==4: return "cv2.INTER_LANCZOS4"
        elif zoom_interpol_method==5: return "cv2.INTER_LANCZOS4"

    if zoom_factor<0.8: #for downsampling the image, all methods perform similar
        #but cv2.INTER_LINEAR, is closest most of the time, irrespective of the zoom_order
        return "cv2.INTER_LINEAR"


def pad_arguments_np2cv(padding_mode):
    """
    NumPy's pad and OpenCVs copyMakeBorder can do the same thing, but the 
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
    str: OpenCV borderType, or "delete" or "alternate"    
        - "cv2.BORDER_CONSTANT": iiiiii|abcdefgh|iiiiiii with some specified i 
        - "cv2.BORDER_REFLECT": fedcba|abcdefgh|hgfedcb
        - "cv2.BORDER_REFLECT_101": gfedcb|abcdefgh|gfedcba
        - "cv2.BORDER_DEFAULT": same as BORDER_REFLECT_101
        - "cv2.BORDER_REPLICATE": aaaaaa|abcdefgh|hhhhhhh
        - "cv2.BORDER_WRAP": cdefgh|abcdefgh|abcdefg
    """
    #Check if padding_mode is already an OpenCV borderType
    padmodes_cv = ["cv2.BORDER_CONSTANT","cv2.BORDER_REFLECT",
                   "cv2.BORDER_REFLECT_101","cv2.BORDER_DEFAULT",
                   "cv2.BORDER_REPLICATE","cv2.BORDER_WRAP"]
    padmodes_cv += ["delete","alternate"]
    #padmodes_cv = [a.lower() for a in padmodes_cv]
    
    #If padding_mode is already one of those, just return the identity
    if padding_mode in padmodes_cv:
        return padding_mode
    
    if "cv2" in padding_mode and "constant" in padding_mode:
        return "cv2.BORDER_CONSTANT"
    elif "cv2" in padding_mode and "replicate" in padding_mode:
        return "cv2.BORDER_REPLICATE"    
    elif "cv2" in padding_mode and "reflect_101" in padding_mode:
        return "cv2.BORDER_REFLECT_101"    
    elif "cv2" in padding_mode and "reflect" in padding_mode:
        return "cv2.BORDER_REFLECT"    
    elif "cv2" in padding_mode and "wrap" in padding_mode:
        return "cv2.BORDER_WRAP" 

    #Check that the padding_mode is actually supported by OpenCV
    supported = ["constant","edge","reflect","symmetric","wrap","delete","alternate"]
    assert padding_mode.lower() in supported, "The padding mode: '"+padding_mode+"' is not supported"
    
    #Otherwise, return the an OpenCV borderType corresponding to the numpy pad mode
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

def image_crop_pad_cv2(images,pos_x,pos_y,pix,final_h,final_w,padding_mode="cv2.BORDER_CONSTANT"):
    """
    Function takes a list images (list of numpy arrays) an resizes them to 
    equal size by center cropping and/or padding.

    Parameters
    ----------
    images: list of images of arbitrary shape
    (nr.images,height,width,channels) 
        can be a single image or multiple images
    pos_x: float or ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [um]
    pos_y: float or ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [um]
        
    final_h: int
        target image height [pixels]
    
    final_w: int
        target image width [pixels]
        
    padding_mode: str; OpenCV BorderType
        Perform the following padding operation if the cell is too far at the 
        border such that the  desired image size cannot be 
        obtained without going beyond the order of the image:
                
        #the following text is copied from 
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5        
        - "cv2.BORDER_CONSTANT": iiiiii|abcdefgh|iiiiiii with some specified i 
        - "cv2.BORDER_REFLECT": fedcba|abcdefgh|hgfedcb
        - "cv2.BORDER_REFLECT_101": gfedcb|abcdefgh|gfedcba
        - "cv2.BORDER_DEFAULT": same as BORDER_REFLECT_101
        - "cv2.BORDER_REPLICATE": aaaaaa|abcdefgh|hhhhhhh
        - "cv2.BORDER_WRAP": cdefgh|abcdefgh|abcdefg

        - "delete": Return empty array (all zero) if the cell is too far at border (delete image)
        - "alternate": randomize the padding operation
    Returns
    ----------
    images: list of images. Each image is a numpy array of shape 
    (final_h,final_w,channels) 

    """
    #Convert position of cell from "um" to "pixel index"
    #pos_x,pos_y = pos_x/pix,pos_y/pix  
    padding_modes = ["cv2.BORDER_CONSTANT","cv2.BORDER_REFLECT","cv2.BORDER_REFLECT_101","cv2.BORDER_REPLICATE","cv2.BORDER_WRAP"]
    
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
        
        #Crop the image
        temp = image[int(y1):int(y2),int(x1):int(x2)]

        if pad_top+pad_bottom+pad_left+pad_right>0:
            if padding_mode.lower()=="delete":
                temp = np.zeros_like(temp)
            else:
                #Perform all padding operations in one go
                if padding_mode.lower()=="alternate":
                    ind = rand_state.randint(low=0,high=len(padding_modes))
                    padding_mode = padding_modes[ind]
                    temp = cv2.copyMakeBorder(temp, pad_top, pad_bottom, pad_left, pad_right, eval(padding_modes[ind]))
                else:
                    temp = cv2.copyMakeBorder(temp, pad_top, pad_bottom, pad_left, pad_right, eval(padding_mode))
        
        images[i] = temp
            
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
    xlsx = pd.ExcelFile(meta_path,engine="openpyxl")
    
    #The zooming factor is saved in the UsedData sheet
    meta = pd.read_excel(xlsx,sheet_name="UsedData",engine="openpyxl")
    zoom_factor = meta["zoom_factor"].iloc[0]#should images be zoomed before forwarding through neural net?

    meta = pd.read_excel(xlsx,sheet_name="Parameters",engine="openpyxl")

    try:
        model_type = meta["Chosen Model"].iloc[0]#input dimensions of the model
    except:
        model_type = "Unknown"
    
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
    img_processing_settings["model_type"]=model_type,
    img_processing_settings["target_imsize"]=target_imsize,
    img_processing_settings["target_channels"]=target_channels,
    img_processing_settings["normalization_method"]=normalization_method,
    img_processing_settings["mean_trainingdata"]=mean_trainingdata,
    img_processing_settings["std_trainingdata"]=std_trainingdata,
    img_processing_settings["zoom_factor"]=zoom_factor,
    img_processing_settings["zoom_interpol_method"]=zoom_interpol_method,
    img_processing_settings["padding_mode"]=padding_mode,
    
    return img_processing_settings

def check_squared(images):
    if images.shape[1]==images.shape[2]:
        return images #everything is fine
    else:
        print("Image is not yet squared. Crop 1 pixel from the longer side to adjust")
        #which is the smaller side?
        if images.shape[1]<images.shape[2]: #height is smaller than width
            images = images[:,:,0:-1]
        elif images.shape[1]>images.shape[2]: #height is smaller than width
            images = images[:,0:-1]
        print("Final size after correcting: "+str(images.shape))
    return images

def gen_crop_img(cropsize,rtdc_path,nr_events=100,replace=True,random_images=True,zoom_factor=1,zoom_order="cv2.INTER_LINEAR",color_mode='Grayscale',padding_mode='constant',xtra_in=False):

    failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
    if failed:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)       
        msg.setText(str(rtdc_ds))
        msg.setWindowTitle("Error occurred during loading file")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        return
    
    pix = rtdc_ds.attrs["imaging:pixel size"] #get pixelation (um/pix)
    #images_shape = rtdc_ds["image"].shape #get shape of the images (nr.images,height,width,channels)
    images = rtdc_ds["events"]["image"] #get the images

    if len(images)<1:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)       
        msg.setText("There are no images")
        msg.setWindowTitle("Empty dataset!")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()       
        return               
    
    if color_mode=='RGB': #User wants RGB images
        channels = 3
    if color_mode=='Grayscale': # User want to have Grayscale
        channels = 1
    
    #Adjust number of channels
    images = image_adjust_channels(images,target_channels=channels)
    
    pos_x,pos_y = rtdc_ds["events"]["pos_x"][:]/pix,rtdc_ds["events"]["pos_y"][:]/pix #/pix converts to pixel index 
    #If there is a zooming to be applied, adjust pos_x and pos_y accordingly
    if zoom_factor != 1:
        pos_x,pos_y = zoom_factor*pos_x,zoom_factor*pos_y
        
    index = list(range(len(pos_x))) #define an index to track, which cells are used from the file
    ind = range(len(images))

    if xtra_in==True:
        xtra_data = np.array(rtdc_ds["xtra_in"])
    if xtra_in==False:
        xtra_data = []#in case xtra_in==None, this empty list will be returned
    
    if random_images==True:
        print("I'm loading random images (from disk)")
        #select a random amount of those cells 
        random_ind = rand_state.choice(ind, size=nr_events, replace=replace) #get random indexes, either unique (replace=False) or not unique (replace=True)                   
        random_ind_unique = np.unique(random_ind,return_counts=True)

        images_required = images[random_ind_unique[0],:,:] #now we have one copy of each image,but some images are required several times
        pos_x,pos_y = pos_x[random_ind_unique[0]],pos_y[random_ind_unique[0]]
        index = np.array(index)[random_ind_unique[0]] 
        
        if xtra_in==True:
            xtra_data = xtra_data[random_ind_unique[0]] 
            
        images,Pos_x,Pos_y,indices,Xtra_data = [],[],[],[],[] #overwrite images by defining the list 'images'
        zoom_interpol_method = zoom_arguments_scipy2cv(zoom_factor,zoom_order)

        for i in range(len(random_ind_unique[1])):
            for j in range(random_ind_unique[1][i]):#when shuffle=True it can happend that some images occure multiple times. This look makes sure this is possible
                images.append(cv2.resize(images_required[i], dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=eval(zoom_interpol_method)))
                Pos_x.append(pos_x[i])
                Pos_y.append(pos_y[i])
                indices.append(index[i])
                if xtra_in==True: 
                    Xtra_data.append(xtra_data[i])
                    

        images = np.array(images)
        pos_x = np.array(Pos_x)
        pos_y = np.array(Pos_y)
        index = np.array(indices)
        if xtra_in==True: 
            xtra_data = np.array(Xtra_data)
 
        permut = np.random.permutation(images.shape[0])
        images = np.take(images,permut,axis=0,out=images) #Shuffle the images
        images = list(images)
        pos_x = np.take(pos_x,permut,axis=0,out=pos_x) #Shuffle pos_x
        pos_y = np.take(pos_y,permut,axis=0,out=pos_y) #Shuffle pos_y
        index = np.take(index,permut,axis=0,out=index) #Shuffle index
        if xtra_in==True:         
            xtra_data = np.take(xtra_data,permut,axis=0,out=xtra_data) #Shuffle xtra_data
        
    if random_images==False:
        print("I'm loading all images (from disk)")
        #simply take all available cells
        random_ind = ind #Here it is NOT a random index, but the index of all cells that are not too close to the image border
        #images = list(np.array(images)[random_ind])
        images = images[:]
        images = list(images)
        zoom_interpol_method = zoom_arguments_scipy2cv(zoom_factor,zoom_order)

        if zoom_factor!=1.0:
            for i in range(len(images)):
                images[i] = cv2.resize(images[i], dsize=None,fx=zoom_factor, fy=zoom_factor, interpolation=eval(zoom_interpol_method))

        pos_x,pos_y = pos_x[random_ind],pos_y[random_ind]
        index = np.array(index)[random_ind] #this is the original index of all used cells
        if xtra_in==True:         
            xtra_data = np.array(xtra_data)[random_ind] #this is the original index of all used cells

    #Cropping and padding operation to obtain images of desired size
    padding_mode = pad_arguments_np2cv(padding_mode)
    images = image_crop_pad_cv2(images=images,pos_x=pos_x,pos_y=pos_y,pix=pix,final_h=cropsize,final_w=cropsize,padding_mode=padding_mode)
    
    images = np.r_[images]
    print("Final size:"+str(images.shape)+","+str(np.array(index).shape))
    #terminate the function by yielding the result
    yield check_squared(images),np.array(index).astype(int),np.array(xtra_data)

def gen_crop_img_ram(dic,rtdc_path,nr_events=100,replace=True,random_images=True,xtra_in=False):        
    Rtdc_path = dic["rtdc_path"]
    ind = np.where(np.array(Rtdc_path)==rtdc_path)[0]
    images = np.array(dic["Cropped_Images"])[ind][0]   
    indices = np.array(dic["Indices"])[ind][0]   
    xtra_data = np.array(dic["Xtra_In"])[ind][0]   

    ind = range(len(images))
    if random_images==True:
        #select a random amount of those cells               
        random_ind = rand_state.choice(ind, size=nr_events, replace=replace) #get random indexes, either unique (replace=False) or not unique (replace=True)                   
        random_ind_unique = np.unique(random_ind,return_counts=True)

        images_required = images[random_ind_unique[0],:,:] #now we have one copy of each image,but some images are required several times
        indices_required = indices[random_ind_unique[0]]
        if xtra_in:
            xtra_data_required = xtra_data[random_ind_unique[0]]
        
        images,indices,xtra_data = [],[],[]
        for i in range(len(random_ind_unique[1])):
            for j in range(random_ind_unique[1][i]):
                images.append(images_required[i,:,:])
                indices.append(indices_required[i])
                if xtra_in:
                    xtra_data.append(xtra_data_required[i])

        images = np.array(images)
        indices = np.array(indices)
        if xtra_in:
            xtra_data = np.array(xtra_data)

        permut = np.random.permutation(images.shape[0])
        images = np.take(images,permut,axis=0,out=images) #Shuffle the images
        indices = np.take(indices,permut,axis=0,out=indices) #Shuffle the images
        if xtra_in:
            xtra_data = np.take(xtra_data,permut,axis=0,out=xtra_data) #Shuffle the images
        
    if random_images==False:
        #simply take all available cells              
        random_ind = ind                   
        images = images
        indices = indices
        xtra_data = xtra_data
        
    yield images,np.array(indices).astype(int),xtra_data

def contrast_augm_numpy(images,fmin,fmax):
    for i in range(images.shape[0]):
        fac = np.random.uniform(low=fmin,high=fmax) #plus minus 15%
        images[i] = np.clip(128 + fac * images[i] - fac * 128, 0, 255).astype(np.uint8)
    return images

def contrast_augm_cv2(images,fmin,fmax):
    """
    this function is equivalent to the numpy version, but 2.8x faster
    """
    images = np.copy(images)
    contr_rnd = rand_state.uniform(low=fmin,high=fmax,size=images.shape[0])
    for i in range(images.shape[0]):
        fac = contr_rnd[i]
        images[i] = np.atleast_3d(cv2.addWeighted(images[i], fac , 0, 0, 128-fac*128))
    return images

def saturation_hue_augm_tf(X_batch,saturation_on,saturation_lower,saturation_higher,hue_on,hue_delta):
    if saturation_on or hue_on:
        g1 = tf.Graph()
        with tf.Graph().as_default() as graph: #Without this multithreading does not work in TensorFlow
            with tf.Session(config=config_gpu) as session:
            #if contrast_on:
            #    X_batch = tf.image.random_contrast(X_batch, contrast_lower, contrast_higher) #Gray and RGB; both values >0!
                if saturation_on:
                    X_batch = tf.image.random_saturation(X_batch, saturation_lower, saturation_higher) #Gray and RGB; both values >0!
                if hue_on:
                    X_batch = tf.image.random_hue(X_batch, hue_delta) #Gray and RGB; both values >0!
                X_batch = X_batch.eval()
    return X_batch.astype(np.uint8)

#def saturation_hue_augm_tf(images,saturation_on,saturation_lower,saturation_higher,hue_on,hue_delta):
#    #################Add multiprocessing power!#######################
#    def random_contrast(x: tf.Tensor) -> tf.Tensor:
#        """Contrast augmentation"""
#        #x = tf.image.random_contrast(x, sat_low, sat_high)
#        if saturation_on:
#            x = tf.image.random_saturation(x, saturation_lower, saturation_higher)
#        if hue_on:
#            x = tf.image.random_hue(x, hue_delta)
#        return x
#    nr_images = images.shape[0]
#    images = tf.data.Dataset.from_tensor_slices(images)#numpy array to tf dataset
#    images = images.map(random_contrast, num_parallel_calls=4)
#    images = images.batch(nr_images)
#    images = [a.numpy() for a in images][0]
#    return images.astype(np.uint8)

def satur_hue_augm_cv2(X_batch,saturation_on,saturation_lower,saturation_higher,hue_on,hue_delta):
    """
    Replacement for the tf version for changing saturation and hue.
    This version is 4.5x faster than saturation_hue_augm_tf
    """
    if saturation_on or hue_on:
        X_batch = np.copy(X_batch)
        hue_delta = abs(hue_delta)
        #get random numbers
        sat_rnd = rand_state.uniform(low=saturation_lower,high=saturation_higher,size=X_batch.shape[0])
        hue_rnd = rand_state.uniform(low=1-hue_delta,high=1+hue_delta,size=X_batch.shape[0])
        for i in range(X_batch.shape[0]):
            hsv = cv2.cvtColor(X_batch[i], cv2.COLOR_RGB2HSV)
            hsv[...,1] = hsv[...,1]*sat_rnd[i] #change the saturation
            hsv[...,0] = hsv[...,0]*hue_rnd[i] #change hue
            X_batch[i] = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return X_batch.astype(np.uint8)

def brightness_noise_augm(images,add_low,add_high,mult_low,mult_high,noise_mean,noise_std):
    m = np.random.uniform(low=mult_low,high=mult_high,size=images.shape[0]) #plus minus 15%
    n = np.random.uniform(low=add_low,high=add_high,size=images.shape[0]) #plus minus 10 Grayscale values
    noise = np.random.normal(loc=noise_mean,scale=noise_std,size=images.shape[1:])
    noise = noise.astype(int)
    images = images.astype(int)
    for i in range(len(images)):
        #np.random.shuffle(noise) #Get newly shuffled noise, this is faster than creating new noise-map
        cv2.randShuffle(noise) #Even faster than numpy shuffle
        images[i] = images[i]*m[i] + n[i] + noise
    images = np.clip(images,0,255)
    return images


def brightn_noise_augm_cv2(images,add_low,add_high,mult_low,mult_high,noise_mean,noise_std):
    """
    this function is equivalent to brightness_noise_augm, but 2x faster
    """
    images = np.copy(images)
    m = rand_state.uniform(low=mult_low,high=mult_high,size=images.shape[0]) #plus minus 15%
    n = rand_state.uniform(low=add_low,high=add_high,size=images.shape[0]) #plus minus 10 Grayscale values
    noise = rand_state.normal(loc=noise_mean,scale=noise_std,size=images.shape[1:])
    noise = noise.astype(np.int16)
    images = images.astype(np.int16)
    buffer = np.zeros(shape=images.shape,dtype=np.uint8)
    for i in range(len(images)):
        cv2.randShuffle(noise) #Even faster than numpy shuffle (actually a LOT!. More than 2x faster!)
        cv2.addWeighted(images[i], m[i] , noise, 1, n[i],dst=buffer[i],dtype=0)
    return buffer

def affine_augm(images,v_flip,h_flip,rot,width_shift,height_shift,zoom,shear):
    """Affine augmentation (replacement for affine augm. for which I previously (AID <=0.0.4) used Keras ImageDataGenerator)
    -Function augments images
    images: array. array of shape (nr.images,image_height,image_width,channels)
    v_flip: bool. If True, 50% of the images are vertically flipped
    h_flip: bool. If True, 50% of the images are horizontally flipped
    rot: integer or float. Range of rotation in degrees
    width_shift: float. If >=1 or <=-1:number of pixels to shift the image left or right, if between 0 and 1: fraction of total width of image
    height_shift: float. If >=1 or <=-1:number of pixels to shift the image up or down, if between 0 and 1: fraction of total height of image
    zoom: float. zoom=0.1 means image is randomly scaled up/down by up to 10%. zoom=10 means, images are randomly scaled up to 10x initial size or down to 10% of initial size
    shear: float. Shear Intensity (Shear angle in degrees)
    
    this functions performs very similar augmentation operations that are also 
    possbile using Keras ImageDataGenerator or Imgaug, but this function is
    7x faster than ImageDataGenerator and
    4.5x faster than Imgaug  
    """
    images = np.copy(images)
    rot,width_shift,height_shift,shear = abs(rot),abs(width_shift),abs(height_shift) ,abs(shear) 
    rows,cols = images.shape[1],images.shape[2]
    if height_shift<1 and height_shift>-1:
        height_shift = height_shift*rows
    if width_shift<1 and width_shift>-1:
        width_shift = width_shift*cols
    
    if zoom!=0: #get the random numbers for zooming
        zoom = abs(zoom)
        if zoom>0 and zoom<1:
            fx = rand_state.uniform(low=1-zoom,high=1+zoom,size=images.shape[0])
            fy = rand_state.uniform(low=1-zoom,high=1+zoom,size=images.shape[0])
        else:
            fx = rand_state.uniform(low=1.0/np.float(zoom),high=zoom,size=images.shape[0])
            fy = rand_state.uniform(low=1.0/np.float(zoom),high=zoom,size=images.shape[0])
    if rot!=0:
        deg_rnd = rand_state.uniform(-rot,rot,size=images.shape[0])
    else:
        deg_rnd = np.repeat(0,repeats=images.shape[0])
    if height_shift!=0:
        height_shift_rnd = rand_state.uniform(-height_shift,height_shift,size=images.shape[0])
    else:
        height_shift_rnd = np.repeat(0,repeats=images.shape[0])
    if width_shift!=0: 
        width_shift_rnd = rand_state.uniform(-width_shift,width_shift,size=images.shape[0])
    else:
        width_shift_rnd = np.repeat(0,repeats=images.shape[0])
    if shear!=0: 
        shear = np.deg2rad(shear)
        shear_rnd = rand_state.uniform(-shear,shear,size=images.shape[0])
    else:
        shear_rnd = np.repeat(0,repeats=images.shape[0])
        
    for i in range(images.shape[0]):
        img = images[i]
        #1. Flipping:
        if v_flip==True and h_flip==False and rand_state.randint(low=0,high=2)>0:
            img = cv2.flip( img, 0 )
        elif v_flip==False and h_flip==True and rand_state.randint(low=0,high=2)>0:
            img = cv2.flip( img, 1 )
        elif v_flip==True and h_flip==True:
            rnd = rand_state.randint(low=-1,high=2) #get a random flipping axis: 1=vertical,0=horizontal,-1=both
            img = cv2.flip( img, rnd )

        #2.zooming
        if zoom!=0:
            img	= np.atleast_3d(cv2.resize(img,None,fx=fx[i],fy=fy[i]))
            #By either padding or cropping, get back to the initial image size
            diff_height,diff_width = img.shape[0]-rows, img.shape[1]-cols
            c_height,c_width = int(img.shape[0]/2), int(img.shape[1]/2)#center in height and width
            #adjust height:
            if diff_height>0:#zoomed image is too high->crop
                y1 = c_height-rows//2
                y2 = y1+rows
                img = img[int(y1):int(y2)]
            if diff_width>0:#zoomed image is too high->crop
                x1 = c_width-cols//2
                x2 = x1+cols
                img = img[:,int(x1):int(x2)]
            
            if diff_height<0 or diff_width<0 :#zoomed image is to small in some direction->pad
                if diff_height<0:
                    diff_height = abs(diff_height)
                    top, bottom = diff_height//2, diff_height-(diff_height//2)
                else:
                    top, bottom = 0,0
                if diff_width<0:
                    diff_width = abs(diff_width)    
                    left, right = diff_width//2, diff_width-(diff_width//2)
                else:
                    left, right = 0,0
                color = [0, 0, 0]
                img = np.atleast_3d(cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color))
        
        #3.Translation, Rotation, Shear
        M_transf = cv2.getRotationMatrix2D((cols/2,rows/2),deg_rnd[i],1) #rotation matrix
        translat_center_x = -(shear_rnd[i]*cols)/2;
        translat_center_y = -(shear_rnd[i]*rows)/2;
        M_transf = M_transf + np.float64([[0,shear_rnd[i],width_shift_rnd[i] + translat_center_x], [shear_rnd[i],0,height_shift_rnd[i] + translat_center_y]]);
        images[i] = np.atleast_3d(cv2.warpAffine(img,M_transf,(cols,rows))) #Rotation, translation and shear in a single call of cv2.warpAffine!
    return images

def avg_blur_cv2(images,k1,k2):
    if k1==k2:
        k = np.zeros(shape=images.shape[0])+k2
    elif k1>k2:
        k = rand_state.randint(low=k2,high=k1,size=images.shape[0])
    else:
        k = rand_state.randint(low=k1,high=k2,size=images.shape[0])

    buffer = np.zeros(shape=images.shape,dtype=np.uint8)
    for i in range(images.shape[0]):
        if k[i]>0:
            cv2.blur(images[i,:,:,:],(k[i],k[i]),dst=buffer[i])
        else:
            buffer[i] = images[i].astype(np.uint8)
    return buffer


def gauss_blur_cv(images,minkernelsize,maxkernelsize):
    #images = np.copy(images)
    if minkernelsize==maxkernelsize:
        k = np.zeros(shape=images.shape[0])+maxkernelsize
    elif minkernelsize>maxkernelsize:
        k = rand_state.randint(low=maxkernelsize,high=minkernelsize,size=images.shape[0])
    else:
        k = rand_state.randint(low=minkernelsize,high=maxkernelsize,size=images.shape[0])
    k = 2*(k//2)+1 #make k odd
    buffer = np.zeros(shape=images.shape,dtype=np.uint8)
    for i in range(images.shape[0]):
        if k[i]>0:
            cv2.GaussianBlur(images[i,:,:,:],(k[i],k[i]),0,dst=buffer[i])
        else:
            buffer[i] = images[i].astype(np.uint8)
    return buffer


def motion_blur_cv(images,kernel_size,angle):
    """
    kernel_size: int or tuple; if int: random kernel sizes from 0 to int are used; if tuple: it has the be (k_min,k_max) which is the minimum and maximum kernel sizes between which random kernel sizes are generated
    angle: int or tuple; if int: only this angle will be used; if tuple: it has to be (angle_min,angle_max) which is the minimum and maximum angle for the direction of motion blur. In RTDC, cells go horizontal-therefore angle=0 makes most sense.
    """    
    if type(kernel_size)==int or type(angle)==float:
        k_rnd = rand_state.randint(low=0,high=kernel_size,size=images.shape[0])
    elif type(kernel_size)==tuple or type(kernel_size)==list or type(kernel_size)==np.ndarray: #and len(kernel_size)==2:
        if np.min(kernel_size)==np.max(kernel_size):
            k_rnd = rand_state.randint(low=0,high=np.min(kernel_size),size=images.shape[0])
        else:
            k_rnd = rand_state.randint(low=np.min(kernel_size),high=np.max(kernel_size),size=images.shape[0])
    else:
        msg = "Range of kernel sizes for motion blurring are wrongly defined"
        raise ValueError(msg)

    if type(angle)==int or type(angle)==float:
        angle_rnd = np.zeros(shape=images.shape[0])+np.min(angle)
    elif type(angle)==tuple or type(angle)==list or type(angle)==np.ndarray: #and len(kernel_size)==2:
        if np.min(angle)==np.max(angle):
            angle_rnd = np.zeros(shape=images.shape[0])+np.min(angle)
        else:
            angle_rnd = rand_state.randint(low=np.min(angle),high=np.max(angle),size=images.shape[0])
    else:
        msg = "Range of angles for motion blurring are wrongly defined"
        raise ValueError(msg)

    buffer = np.zeros(shape=images.shape,dtype=np.uint8)
    for i in range(images.shape[0]):
        if k_rnd[i]>0:
            #Source of equation:iperov from Stackoverflow (Thanks!!!): https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
            # generating the kernel
            k = np.zeros((k_rnd[i], k_rnd[i]),dtype=np.float32)
            k[ (k_rnd[i]-1)//2,:] = np.ones(k_rnd[i],dtype=np.float32)
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (k_rnd[i]/2-0.5,k_rnd[i]/2-0.5),angle_rnd[i],1.0),(k_rnd[i],k_rnd[i]) )  
            k = k*(1.0/np.sum(k))        
            cv2.filter2D(images[i,:,:,:],-1,k,dst=buffer[i])
        else:
            buffer[i] = images[i]
    return buffer


def crop_imgs_to_ram(SelectedFiles,crop,zoom_factors=None,zoom_order=0,color_mode='Grayscale'):
    #This function transfers the entire data to ram which allows to access it from there 
    #quickly. This makes lots of sense for most data sets, because the cropped images with uint8 take very little space.
    #Compute estimate of required disk space using the function print_ram_example if you like
    
    Rtdc_paths = [selectedfile["rtdc_path"] for selectedfile in SelectedFiles] #get rtdc paths
    Rtdc_paths_uni = np.unique(np.array(Rtdc_paths)) #get unique Rtdc_paths
    xtra_in = set([selectedfile["xtra_in"] for selectedfile in SelectedFiles])
    if len(xtra_in)>1:# False and True is present. Not supported
        print("Xtra data is used only for some files. Xtra data needs to be used either by all or by none!")
        return
    xtra_in = list(xtra_in)[0]#this is either True or False

    X_train,Indices,Xtra_in_data = [],[],[]
    for i in range(len(Rtdc_paths_uni)): #Move all images to RAM (Not only some random images!)->random_images=False
        if zoom_factors!=None:
            gen_train = gen_crop_img(crop,Rtdc_paths_uni[i],random_images=False,zoom_factor=zoom_factors[i],zoom_order=zoom_order,color_mode=color_mode,xtra_in=xtra_in) #Replace=true means that individual cells could occur several times    
        else:
            gen_train = gen_crop_img(crop,Rtdc_paths_uni[i],random_images=False,color_mode=color_mode,xtra_in=xtra_in) #Replace=true means that individual cells could occur several times    
        
        x_train,index,xtra_in_data = next(gen_train)        
        X_train.append(x_train)
        Indices.append(index)
        Xtra_in_data.append(xtra_in_data)
        
    dic = {"rtdc_path":Rtdc_paths_uni,"Cropped_Images":X_train,"Indices":Indices,"Xtra_In":Xtra_in_data}
    return dic

def image_normalization(images,normalization_method,mean_trainingdata=None,std_trainingdata=None):
    """
    Perform a normalization of the pixel values.

    Parameters
    ----------
    images: ndarray
    normalization_method: str
        Factor by which the size of the images should be zoomed
    normalization_method: str; available are: (text copied from original docs: 
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize)
        -"None" – No normalization is applied.
        -"Div. by 255" – Each input image is divided by 255 (useful since pixel 
        values go from 0 to 255, so the result will be in range 0-1)
        -"StdScaling using mean and std of each image individually" – The mean 
        and standard deviation of each input image itself is used to scale it 
        by first subtracting the mean and then dividing by the standard deviation
        -"StdScaling using mean and std of all training data" - During model 
        training, the mean and std of the entire training set was determined. 
        This mean and standard deviation is used to normalize images by first 
        subtracting the mean and then dividing by the standard deviation    
    mean_trainingdata: float; the mean pixel value obtained from the training dataset
    std_trainingdata: float; the std of the pixel values obtained from the training dataset

    Returns
    ----------
    ndarray of images
  
    """
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


def imgs_2_rtdc(fname,images,pos_x,pos_y):
    #There needs to be an empty .rtdc file which will be copied
    shutil.copy(os.path.join(dir_root,"Empty.rtdc"),fname)
    if len(images.shape)==4:    
        maxshape = (None, images.shape[1], images.shape[2], images.shape[3])
    elif len(images.shape)==3:    
        maxshape = (None, images.shape[1], images.shape[2])

    #Create rtdc_dataset
    hdf = h5py.File(fname,'a')
    hdf.create_dataset("events/image", data=images, dtype=np.uint8,maxshape=maxshape,fletcher32=True,chunks=True)
    hdf.create_dataset("events/pos_x", data=pos_x, dtype='uint8')
    hdf.create_dataset("events/pos_y", data=pos_y, dtype='uint8')
    hdf.create_dataset("events/index", data=np.array(range(len(pos_x)))+1, dtype='uint8')
    
    #Adjust metadata:
    #"experiment:event count" = Nr. of images
    hdf.attrs["experiment:event count"]=images.shape[0]
    hdf.attrs["experiment:sample"]=fname
    hdf.attrs["imaging:pixel size"]=1.00
    hdf.attrs["experiment:event count"]=images.shape[0]
    hdf.close()


#def image_resize_crop_pad(images,pos_x,pos_y,final_h,final_w,channels,verbose=False,padding_mode='constant'):
#    """
#    Function takes a list images (list of numpy arrays) an resizes them to 
#    an equal size by center cropping and/or padding.
#    
#    images: list of images of arbitrary shape
#    final_h: integer, defines final height of the images
#    final_w: integer, defines final width of the images
#    """
#    for i in range(len(images)):
#        image = images[i]
#        #HEIGHT
#        diff_h = int(abs(final_h-image.shape[0]))
#        #Padding or Cropping?
#        if final_h > image.shape[0]: #Cropsize is larger than the image_shape
#            padding_h = True #if the requested image height is larger than the zoomed in version of the original images, I have to pad
#            diff_h = int(np.round(abs(final_h-image.shape[0])/2.0,0))
#            if verbose:
#                print("Padding height: "+str(diff_h) + " pixels each side")
#        elif final_h <= image.shape[0]:
#            padding_h = False
#            diff_h = int(np.round(abs(final_h-image.shape[0])/2.0,0))
#            if verbose:
#                print("Cropping height: "+str(diff_h) + " pixels each side")
#    
#        #WIDTH
#        diff_w = int(abs(final_w-image.shape[1]))
#        #Padding or Cropping?
#        if final_w > image.shape[1]: #Cropsize is larger than the image_shape
#            padding_w = True #if the requested image height is larger than the zoomed in version of the original images, I have to pad
#            diff_w = int(np.round(abs(final_w-image.shape[1])/2.0,0))
#            if verbose:
#                print("Padding width: "+str(diff_w) + " pixels each side")
#        elif final_w <= image.shape[1]:
#            padding_w = False
#            diff_w = int(np.round(abs(final_w-image.shape[1])/2.0,0))
#            if verbose:
#                print("Cropping width: "+str(diff_w) + " pixels each side")
#        #In case of odd image shapes:
#        #check if the resulting cropping or padding operation would return the correct size
#        odd_h,odd_w = -1,-1
#        if padding_w == True:
#            while final_w!=image.shape[1]+2*diff_w+odd_w:
#                odd_w+=1 #odd_w is increased until the resulting image is of correct size
#        elif padding_w == False:
#            while final_w!=image.shape[1]-2*diff_w+odd_w:
#                odd_w+=1 #odd_w is increased until the resulting image is of correct size
#        if padding_h == True:
#            while final_h!=image.shape[0]+2*diff_h+odd_h:
#                odd_h+=1 #odd_w is increased until the resulting image is of correct size
#        elif padding_h == False:
#            while final_h!=image.shape[0]-2*diff_h+odd_h:
#                odd_h+=1 #odd_w is increased until the resulting image is of correct size
#                
#        #Execute padding-only operation and overwrite original on list "images"
#        if padding_h==True and padding_w==True:
#            if channels==1:
#                images[i] = np.pad(image,pad_width=( (diff_h, diff_h+odd_h),(diff_w, diff_w+odd_w) ),mode=padding_mode)
#            elif channels==3:
#                images[i] = np.pad(image,pad_width=( (diff_h, diff_h+odd_h),(diff_w, diff_w+odd_w),(0, 0) ),mode=padding_mode)
#            else:
#                if verbose:
#                    print("Invalid image dimensions: "+str(image.shape))
#    
#        #Execute cropping-only operation and overwrite original on list "images"
#        elif padding_h==False and padding_w==False:
#            #Compute again the x,y locations of the cells (this is fast)
#            y1 = np.around(pos_y[i]-final_h/2.0)              
#            x1 = np.around(pos_x[i]-final_w/2.0) 
#            y2 = y1+final_h               
#            x2 = x1+final_w
#            #overwrite the original image
#            images[i] = image[int(y1):int(y2),int(x1):int(x2)]
#            
#        else:
#            if padding_h==True:
#                if channels==1:
#                    image = np.pad(image,pad_width=( (diff_h, diff_h+odd_h),(0, 0) ),mode=padding_mode)
#                elif channels==3:
#                    image = np.pad(image,pad_width=( (diff_h, diff_h+odd_h),(0, 0),(0, 0) ),mode=padding_mode)
#                else:
#                    if verbose:
#                        print("Invalid image dimensions: "+str(image.shape))
#                if verbose:
#                    print("Image size after padding heigth :"+str(image.shape))
#                
#            if padding_w==True:
#                if channels==1:
#                    image = np.pad(image,pad_width=( (0, 0),(diff_w, diff_w+odd_w) ),mode=padding_mode)
#                elif channels==3:
#                    image = np.pad(image,pad_width=( (0, 0),(diff_w, diff_w+odd_w),(0, 0) ),mode=padding_mode)
#                else:
#                    if verbose:
#                        print("Invalid image dimensions: "+str(image.shape))
#                if verbose:
#                    print("Image size after padding width :"+str(image.shape))
#        
#            if padding_h==False:
#                #Compute again the x,y locations of the cells (this is fast)
#                y1 = np.around(pos_y[i]-final_h/2.0)              
#                y2 = y1+final_h               
#                image = image[int(y1):int(y2),:]
#                if verbose:
#                    print("Image size after cropping height:"+str(image.shape))
#        
#            if padding_w==False:
#                #Compute again the x,y locations of the cells (this is fast)
#                x1 = np.around(pos_x[i]-final_w/2.0) 
#                x2 = x1+final_w
#                image = image[:,int(x1):int(x2)]
#                if verbose:
#                    print("Image size after cropping width:"+str(image.shape))
#            
#            images[i] = image
#    return images


def image_resize_scale(images,pos_x,pos_y,final_h,final_w,channels,interpolation_method,verbose=False):
    """
    Function takes a list images (list of numpy arrays) an resizes them to 
    an equal size by scaling (interpolation).
    
    images: list of images of arbitrary shape
    final_h: integer, defines final height of the images
    final_w: integer, defines final width of the images
    interpolation_method: available are: (text copied from original docs: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize)
        -cv2.INTER_NEAREST – a nearest-neighbor interpolation
        -cv2.INTER_LINEAR – a bilinear interpolation (used by default)
        -cv2.INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        -cv2.INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        -cv2.INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    """
    if interpolation_method == "Nearest":
        interpolation_method = cv2.INTER_NEAREST
    if interpolation_method == "Linear":
        interpolation_method = cv2.INTER_LINEAR
    if interpolation_method == "Area":
        interpolation_method = cv2.INTER_AREA
    if interpolation_method == "Cubic":
        interpolation_method = cv2.INTER_CUBIC
    if interpolation_method == "Lanczos":
        interpolation_method = cv2.INTER_LANCZOS4
    for i in range(len(images)):
        #the order width,height in cv2.resize is not an error. OpenCV wants this order...
        images[i] = cv2.resize(images[i], dsize=(final_w,final_h), interpolation=interpolation_method)
    return images


def clip_contrast(img,low,high,auto=False):
    if auto==True:
        low,high = np.min(img),np.max(img)
    # limit_lower = limits[0]
    # limit_upper = limits[1]
    img[:,:] = np.clip(img[:,:],a_min=low,a_max=high)
    mini,maxi = np.min(img[:,:]),np.max(img[:,:])/255
    img[:,:] = (img[:,:]-mini)/maxi
    return img

# def stretch_contrast(img):
#     mini,maxi = np.min(img),np.max(img)
#     factor = 255/maxi
#     img = img.astype(np.float32)
#     img = (img-mini)*factor
#     img = img.astype(np.uint8)
#     return img
    










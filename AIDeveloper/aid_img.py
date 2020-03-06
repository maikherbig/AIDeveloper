# -*- coding: utf-8 -*-
"""
aid_img
some functions for image processing that are essential for AIDeveloper
---------
@author: maikherbig
"""

import numpy as np
import os, shutil,h5py
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import aid_bin
from pyqtgraph.Qt import QtWidgets
from scipy import ndimage
import cv2
import tensorflow as tf
from tensorflow.python.client import device_lib
device_types = device_lib.list_local_devices()
device_types = [device_types[i].device_type for i in range(len(device_types))]
config_gpu = tf.ConfigProto()
if device_types[0]=='GPU':
    config_gpu.gpu_options.allow_growth = True
    config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.7

dir_root = os.path.dirname(aid_bin.__file__)#ask the module for its origin

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

def gen_crop_img(cropsize,rtdc_path,nr_events=100,replace=True,random_images=True,zoom_factor=1,zoom_order=0,color_mode='Grayscale'):
    failed,rtdc_ds = aid_bin.load_rtdc(rtdc_path)
    if failed:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)       
        msg.setText(str(rtdc_ds))
        msg.setWindowTitle("Error occurred during loading file")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        return
    
    pix = rtdc_ds.config["imaging"]["pixel size"] #get pixelation (um/pix)
    images_shape = rtdc_ds["image"].shape #get shape of the images (nr.images,height,width,channels)
    images = rtdc_ds["image"] #get the images
    
    if len(images_shape)==4:#Loaded images are RGB
        if images_shape[-1]==3:
            channels=3
        else:
            print("Images have "+str(images_shape[-1])+" channels. This is (currently) not supported by AID")
            return
        if color_mode=='Grayscale':#User want to have Grayscale: use the luminosity formula to convert RGB to gray
            print("Used luminosity formula to convert RGB to Grayscale")
            images = (0.21 * images[:,:,:,:1]) + (0.72 * images[:,:,:,1:2]) + (0.07 * images[:,:,:,-1:])
            images = images[:,:,:,0] 
            images  = images.astype(np.uint8)           
            channels=1
            
    elif len(images_shape)==3:#Loaded images are Grayscale
        channels=1
        if color_mode=='RGB':#If the user want to use RGB, but did only load Grayscale images, simply copy the information to all 3 channels
            images = np.stack((images,)*3, axis=-1)
            print("Copied information to all three channels to convert Grayscale to RGB")

            channels = 3 #Updates the channel-info. After the conversion we now have RGB

    #HEIGHT
    #Compute, if after zooming, the image would need to be cropped or padded in height
    #Difference between the (zoomed) image height and the required final height?
    diff_h = int(abs(cropsize-zoom_factor*images_shape[1]))
    #Padding or Cropping?
    if cropsize > zoom_factor*images_shape[1]: #Cropsize is larger than the image_shape
        padding_h = True #if the requested image height is larger than the zoomed in version of the original images, I have to pad
        diff_h = int(np.round(abs(cropsize-zoom_factor*images_shape[1])/2.0,0))
        print("I will pad in height: "+str(diff_h) + " pixels on each side")
    elif cropsize <= zoom_factor*images_shape[1]:
        padding_h = False
        diff_h = int(np.round(abs(cropsize-zoom_factor*images_shape[1])/2.0,0))
        print("I will crop: "+str(diff_h) + " pixels in height")

    #WIDTH
    #Compute, if after zooming, the image would need to be cropped or padded in width
    #Difference between the (zoomed) image width and the required final width?
    diff_w = int(abs(cropsize-zoom_factor*images_shape[2]))
    #Padding or Cropping?
    if cropsize > zoom_factor*images_shape[2]: #Cropsize is larger than the image_shape
        padding_w = True #if the requested image height is larger than the zoomed in version of the original images, I have to pad
        diff_w = int(np.round(abs(cropsize-zoom_factor*images_shape[2])/2.0,0))
        print("I will pad in width: "+str(diff_w) + " pixels on each side")
    elif cropsize <= zoom_factor*images_shape[2]:
        padding_w = False
        diff_w = int(np.round(abs(cropsize-zoom_factor*images_shape[2])/2.0,0))
        print("I will crop: "+str(diff_h) + " pixels in width")

    pos_x,pos_y = rtdc_ds["pos_x"][:]/pix,rtdc_ds["pos_y"][:]/pix #/pix converts to pixel index 
    #If there is a zooming to be applied, adjust pos_x and pos_y accordingly
    if zoom_factor != 1:
        pos_x,pos_y = zoom_factor*pos_x,zoom_factor*pos_y
        
    index = list(range(len(pos_x))) #define an index to track, which cells are used from the file
    y1 = np.around(pos_y-cropsize/2.0)              
    x1 = np.around(pos_x-cropsize/2.0) 
    y2 = y1+cropsize                
    x2 = x1+cropsize

    if padding_w==False: #If there is no padding in width, means cells that are at the border can be out of frame
        #Indices of cells that would fit into the required cropping frame (cells at the end of the image do not fit)
        ind = np.where( (x1>=0) & (x2<=zoom_factor*images_shape[2]) & (y1>=0) & (y2<=zoom_factor*images_shape[1]))[0]        
    if padding_w==True:
        ind = range(len(images))

    if random_images==True:
        print("I'm loading random images (from disk)")
        #select a random amount of those cells 
        if len(ind)<1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)       
            msg.setText("Discarded all events because too far at border of image (check zooming/cropping settings!)")
            msg.setWindowTitle("Empty dataset!")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()       
            return               

        random_ind = rand_state.choice(ind, size=nr_events, replace=replace) #get random indexes, either unique (replace=False) or not unique (replace=True)                   
        random_ind_unique = np.unique(random_ind,return_counts=True)

        images_required = images[random_ind_unique[0],:,:] #now we have one copy of each image,but some images are required several times
        pos_x,pos_y = pos_x[random_ind_unique[0]],pos_y[random_ind_unique[0]]
        index = np.array(index)[random_ind_unique[0]] 

        images,Pos_x,Pos_y,indices = [],[],[],[] #overwrite images by defining the list images
        for i in range(len(random_ind_unique[1])):
            for j in range(random_ind_unique[1][i]):
                if channels==1:
                    images.append(ndimage.zoom(images_required[i,:,:], zoom=zoom_factor,order=int(zoom_order)))
                elif channels==3:
                    images.append(ndimage.zoom(images_required[i,:,:], zoom=(zoom_factor,zoom_factor,1),order=int(zoom_order)))

                Pos_x.append(pos_x[i])
                Pos_y.append(pos_y[i])
                indices.append(index[i])

        images = np.array(images)
        pos_x = np.array(Pos_x)
        pos_y = np.array(Pos_y)
        index = np.array(indices)

        permut = np.random.permutation(images.shape[0])
        images = np.take(images,permut,axis=0,out=images) #Shuffle the images
        pos_x = np.take(pos_x,permut,axis=0,out=pos_x) #Shuffle pos_x
        pos_y = np.take(pos_y,permut,axis=0,out=pos_y) #Shuffle pos_y
        index = np.take(index,permut,axis=0,out=index) #Shuffle index
        
    if random_images==False:
        print("I'm loading all images (from disk)")
        #simply take all available cells
        random_ind = ind #Here it is NOT a random index, but the index of all cells that are not too close to the image border
        if len(ind)<1:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)       
            msg.setText("Discarded all events because too far at border of image (check zooming/cropping settings!)")
            msg.setWindowTitle("Empty dataset!")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()       
            return               

        images = np.array(images)[random_ind]
        
        if channels==1:                
            images = ndimage.zoom(images, zoom=(1,zoom_factor,zoom_factor),order=int(zoom_order))
        elif channels==3:                
            images = ndimage.zoom(images, zoom=(1,zoom_factor,zoom_factor,1),order=int(zoom_order))

        pos_x,pos_y = pos_x[random_ind],pos_y[random_ind]
        index = np.array(index)[random_ind] #this is the original index of all used cells

    if padding_h==True and padding_w==True:
        if channels==1:
            images = np.pad(images,pad_width=( (0, 0),(diff_h, diff_h),(diff_w, diff_w) ),mode='constant')
        elif channels==3:
            images = np.pad(images,pad_width=( (0, 0),(diff_h, diff_h),(diff_w, diff_w),(0, 0) ),mode='constant')
        else:
            print("Invalid image dimensions: "+str(images.shape))
            return
        print("Final size:"+str(images.shape)+","+str(np.array(index).shape))
        #terminate the function by yielding the result
        yield check_squared(images),np.array(index).astype(int)

    if padding_h==False and padding_w==False:
        #Compute again the x,y locations of the cells (this is fast)
        y1 = np.around(pos_y-cropsize/2.0)              
        x1 = np.around(pos_x-cropsize/2.0) 
        y2 = y1+cropsize                
        x2 = x1+cropsize
        Images_Cropped = []
        for j in range(len(x2)):#crop the images
            image_cropped = images[j,int(y1[j]):int(y2[j]),int(x1[j]):int(x2[j])]
            #if image_cropped.shape==(cropsize,cropsize):
            Images_Cropped.append(image_cropped)
        images = np.r_[Images_Cropped]
        print("Final size:"+str(images.shape)+","+str(np.array(index).shape))
        #terminate the function by yielding the result
        yield check_squared(images),np.array(index).astype(int)

    if padding_h==True:
        if channels==1:
            images = np.pad(images,pad_width=( (0, 0),(diff_h, diff_h),(0, 0) ),mode='constant')
        elif channels==3:
            images = np.pad(images,pad_width=( (0, 0),(diff_h, diff_h),(0, 0),(0, 0) ),mode='constant')
        else:
            print("Invalid image dimensions: "+str(images.shape))
            return
        print("Image size after padding heigth :"+str(images.shape)+","+str(np.array(index).shape))
        #dont yield here since cropping width could still be required
        
    if padding_w==True:
        if channels==1:
            images = np.pad(images,pad_width=( (0, 0),(0, 0),(diff_w, diff_w) ),mode='constant')
        elif channels==3:
            images = np.pad(images,pad_width=( (0, 0),(0, 0),(diff_w, diff_w),(0, 0) ),mode='constant')
        else:
            print("Invalid image dimensions: "+str(images.shape))
            return
        print("Image size after padding width :"+str(images.shape)+","+str(np.array(index).shape))
        #dont yield here since cropping height could still be required

    if padding_h==False:
        #Compute again the x,y locations of the cells (this is fast)
        y1 = np.around(pos_y-cropsize/2.0)              
        y2 = y1+cropsize                
        Images_Cropped = []
        for j in range(len(y1)):#crop the images
            image_cropped = images[j,int(y1[j]):int(y2[j]),:]
            Images_Cropped.append(image_cropped)
        images = np.r_[Images_Cropped]           
        print("Image size after cropping height:"+str(images.shape)+","+str(np.array(index).shape))

    if padding_w==False:
        #Compute again the x,y locations of the cells (this is fast)
        x1 = np.around(pos_x-cropsize/2.0) 
        x2 = x1+cropsize
        Images_Cropped = []
        for j in range(len(x2)):#crop the images
            image_cropped = images[j,:,int(x1[j]):int(x2[j])]
            Images_Cropped.append(image_cropped)
        images = np.r_[Images_Cropped]           
        print("Image size after cropping width:"+str(images.shape)+","+str(np.array(index).shape))

    print("Final size:"+str(images.shape)+","+str(np.array(index).shape))
    yield check_squared(images),np.array(index).astype(int)

def gen_crop_img_ram(dic,rtdc_path,nr_events=100,replace=True,random_images=True):        
    Rtdc_path = dic["rtdc_path"]
    ind = np.where(np.array(Rtdc_path)==rtdc_path)[0]
    images = np.array(dic["Cropped_Images"])[ind][0]   
    indices = np.array(dic["Indices"])[ind][0]   

    ind = range(len(images))
    if random_images==True:
        #select a random amount of those cells               
        random_ind = rand_state.choice(ind, size=nr_events, replace=replace) #get random indexes, either unique (replace=False) or not unique (replace=True)                   
        random_ind_unique = np.unique(random_ind,return_counts=True)

        images_required = images[random_ind_unique[0],:,:] #now we have one copy of each image,but some images are required several times
        indices_required = indices[random_ind_unique[0]]
        
        images,indices = [],[]
        for i in range(len(random_ind_unique[1])):
            for j in range(random_ind_unique[1][i]):
                images.append(images_required[i,:,:])
                indices.append(indices_required[i])

        images = np.array(images)
        indices = np.array(indices)

        permut = np.random.permutation(images.shape[0])
        images = np.take(images,permut,axis=0,out=images) #Shuffle the images
        indices = np.take(indices,permut,axis=0,out=indices) #Shuffle the images
        
    if random_images==False:
        #simply take all available cells              
        random_ind = ind                   
        images = images
        indices = indices
        
    yield images,np.array(indices).astype(int)

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
    
    X_train,Indices = [],[]
    for i in range(len(Rtdc_paths_uni)): #Move all images to RAM (Not only some random images!)->random_images=False
        if zoom_factors!=None:
            gen_train = gen_crop_img(crop,Rtdc_paths_uni[i],random_images=False,zoom_factor=zoom_factors[i],zoom_order=zoom_order,color_mode=color_mode) #Replace=true means that individual cells could occur several times    
        else:
            gen_train = gen_crop_img(crop,Rtdc_paths_uni[i],random_images=False,color_mode=color_mode) #Replace=true means that individual cells could occur several times    
        
        x_train,index = next(gen_train)        
        X_train.append(x_train)
        Indices.append(index)        
    dic = {"rtdc_path":Rtdc_paths_uni,"Cropped_Images":X_train,"Indices":Indices}
    return dic


def norm_imgs(X,norm,mean_trainingdata=None,std_trainingdata=None):
    if norm == "StdScaling using mean and std of all training data":
        if np.allclose(std_trainingdata,0):
            std_trainingdata = 0.0001
            print("Set the standard deviation (std_trainingdata) to 0.0001 because otherwise div. by 0 would have happend!")

    if len(X.shape)==3: #single channel Grayscale rtdc data
        #Add the "channels" dimension
        X = np.expand_dims(X,3)    
    X = X.astype(np.float32)
    for k in range(X.shape[0]):
        line = X[k,:,:,:]
        ###########Scaling############
        if norm == "None":
            pass
        elif norm == "Div. by 255":
            line = line/255.0
        elif norm == "StdScaling using mean and std of each image individually":
            mean = np.mean(line)
            std = np.std(line)
            if np.allclose(std,0):
                std = 0.0001
                print("Set the standard deviation to 0.0001 because otherwise div. by 0 would have happend!")
            line = (line-mean)/std
        elif norm == "StdScaling using mean and std of all training data":
            line = (line-mean_trainingdata)/std_trainingdata 
        #Under NO circumstances, training data should contain nan values
        ind = np.isnan(line)
        line[ind] = np.random.random() #replace nan with random values. This is better than nan, since .fit will collapse and never get back
        X[k,:,:,:] = line   
    return X        

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




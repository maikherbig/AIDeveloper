#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
cif to rtdc converter
This module reads image data from compensated image files (.cif files) 
and converts the file to hdf5 (.rtdc), compatible to AIDeveloper
The cif files are generated by imaging flow cytometry (IFC) instruments Amnis ImageStream

dependencies: 
install Java development kit (JDK)
Windows installer can be found here
https://www.oracle.com/java/technologies/downloads/#jdk18-windows
Direct download link for windows 
https://download.oracle.com/java/18/latest/jdk-18_windows-x64_bin.exe
 
Next, generate a Python environment (e.g. using conda) using following commands

conda create -n imagestream1 python==3.9 spyder==5
conda activate imagestream1
pip install javabridge==1.0.19
pip install python-bioformats==4.0.5
pip install pandas==1.1.5
pip install opencv-contrib-python-headless==4.5.5.62
pip install openpyxl==3.0.9
pip install xlrd==2.0.1
pip install matplotlib==3.5.2
pip install joblib==1.1.0
pip install tqdm==4.64.0
pip install h5py==3.7.0   

@author: Maik Herbig 2022
"""

import bioformats
import javabridge
import numpy as np
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import cv2
import matplotlib.pyplot as plt
import time
import h5py
import pandas as pd

#provide a path to the cif file
path_cif = r"D:\UTokyo-WORK\07_Own_Projects\08_Amnis_Imagestream\01_Data\20220614_Cells.cif"
path_rtdc = path_cif.replace(".cif","_new2.rtdc")
print(f"Target file name: {path_rtdc}")
sample_name = "SSM cells"

#final image size. If you are unsure, use show_exmple_cell to get the image size of the original images. 
image_size = 80 # number of pixels of output image

# define, which channels you want to extract. AIDeveloper (currently supports up to 3 channels)
# Options (all compatible with AIDeveloper):
    # select one channel (e.g. channels = [5]) - returns grayscale images (also compatible with ShapeOut)
    # select two channels (e.g. channels = [5,8]) - returns RGB images, the last channel is all zeros. This "RGB: image set is not compatible with ShapeOut
    # select three channels (e.g. channels = [0,5,8]) - returns RGG images. Not compatible with ShapeOut.
# if you are unsure which channels you want, make use of the function show_exmple_cell to view some example images
channels = [0,2,4]#[0,5,8] #inidcate which channels to use for [R,G,B]

# Define the upper limit of the fluorence intensities. Needed for conversion to uint8
# Please see bottom of script how to use "intensity_range_info(cif_sample)", which will help
channel_fluo_upper_limit = [800,800,3000] # range of the fluorescence intensities of each channel. Does not need to be the maximum fl intensity
channel_scaling_factors = [255/x for x in channel_fluo_upper_limit] # multiply by 255 since final images should be uint8 (0-255)

# Info about availability of masks. Some ImageStream data stoes the masks in 
# odd series-numbers. Plot the images of a series by calling 
#   show_exmple_cell(series_index=0) #plots fluorescence images of all channels
#   show_exmple_cell(series_index=1) #plots mask images of all channels
mask_available = True
mask_channel = 0 #which channel's mask should be used. You can only use one channel!

# in case, multiple objects are found in an image, only one of them will be selected 
# to compute features. Please indicate if the most "smooth" (minimum area_ratio) 
# or the most "centered" objects should be taken
contour_selection = "centered" # can be "smooth" or "centered"

# how many images should be read from the cif file? Use np.inf if you want to read all.
number_cells = np.inf 

pixel_size = 0.5 # optional, (only used when masks are available). Unit [um/pix]; ImageStreamX MkII has 1.0, 0.5, or 0.3 um/pix for 20x, 40x, and 60x magnification, respectively

###############################################################################
###############################END OF USER INPUT###############################
###############################################################################

# start a java virtual machine
javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='8G')

# initialize a reader for cif file
print("Initilize a reader. That can take a minute")
reader = bioformats.formatreader.get_image_reader("tmp", path=path_cif)

series_count = reader.rdr.getSeriesCount()
series_count = int(series_count/2) #half of them are masks
channel_count = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)
print("\n\n\nReader initialized. File properties:")
print(f"ESTIMATE Nr of series (a series is a collection of images (multichannel) for one event): {series_count}  ")
print(f"Nr of channels: {channel_count}")

def show_cell_from_cif(series_index):
    #show all channels of a cell
    print(f"Show example image. Use image index {series_index}")
    image = reader.read(c=None, series=series_index,rescale=False)
    for ch in range(channel_count):
        plt.figure(1)
        img = image[:,:,ch]
        h,w = image.shape[0],image.shape[1]
        mini,maxi = np.min(img), np.max(img)
        plt.imshow(img,cmap="gray")
        plt.title(f"Ch. {ch}, min={mini}, max={maxi}, w={w}, h={h}")
        plt.show()

def show_cell_from_array(cif_data):
    print("Plotting...")
    #show all channels of a cell
    for index in range(len(cif_data["images"])):
        image = cif_data["images"][index,...]
        plt.figure(1)
        h,w = image.shape[0],image.shape[1]
        mini,maxi = np.min(image), np.max(image)
        plt.imshow(image,cmap="gray")
        plt.title(f"min={mini}, max={maxi}, w={w}, h={h}")
        plt.show()


def uint16_to_uint8(img16,channel_scaling_factor):
    img16 = img16.astype(np.float32)
    img8 = np.multiply(img16, channel_scaling_factor)
    img8 = np.clip(img8,a_min=0,a_max=255)
    img8 = img8.astype(np.uint8)
    return img8

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
    pos_x = [pos_x_/pix for pos_x_ in pos_x]
    pos_y = [pos_y_/pix for pos_y_ in pos_y]
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

def get_brightness(img,mask):
    """
    Compute brightness features from the image (pixels inside contour)

    Parameters
    ----------
    img : numpy array of dimensions (width,height)
        Grayscale image.
    mask : numpy array of dimensions (width,height)
        Binary mask.

    Returns
    -------
    dictinary with following entries:
        
    bright_avg : float
        Mean grayscale value of pixels inside the contour.
    bright_sd : float
        Standard deviation of the grayscale values of pixels inside the contour.
    """

    bright = cv2.meanStdDev(img, mask=mask)
    return {"bright_avg":bright[0][0,0],"bright_sd":bright[1][0,0]}

def get_contour_features(mask,selectcell="centered"):
    """
    Perform contour finding based on masks of the (possibly) multiple cells.
    Choose one single object and compute features (size, circularity)

    Parameters
    ----------
    mask: binary numpy array of dimensions (width,height)
        Binary mask image.
    selectcell : str
        When multiple contours are found, which should be preferred?
        Options: 
            "smooth" - contours which are smooth will be returned
            "centered" - contours with small distance to middle of image will be returned

    Returns
    -------
    dictinary with following entries:
    mask: nd array (binary)
        The binary mask of the chosen object. Can be different from the input mask, since only a single contour is selected.
    area_orig : float
        Size of the original contour [pixels]
    area_hull : float
        Size of the convex hull of the object [pixels].
    area_ratio : float
        area_hull/area_orig. Feature describes 'smoothness' of the contour
    circularity : float
        describes much the shape resebles a circle. Perfect circle has circularity=1. Circularity get lower (towards 0) for non-circular shapes
    """
    
    #binarize image (everything above 0 becomes 1)
    mask = np.clip(mask,a_min=0,a_max=1)

    #for contours, dont use RETR_TREE, but RETR_EXTERNAL as we are not interested in internal objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    
    #in case there is no contour found, add a dummy contour
    if len(contours)==0:
        contours = [np.array([[[0, 0]],[[0, 1]],[[1, 1]],[[1, 0]]])] #generate a dummy contour

    #Sort contours, longest first
    contours.sort(key=len,reverse=True)
    contours = [c for c in contours if len(c)>4] #proper contour should have at least 5 points
    hulls = [cv2.convexHull(contour,returnPoints=True) for contour in contours]

    mu_origs = [cv2.moments(contour) for contour in contours]
    mu_hulls = [cv2.moments(hull) for hull in hulls]

    area_origs = [mu_orig["m00"] for mu_orig in mu_origs]
    area_hulls = [mu_hull["m00"] for mu_hull in mu_hulls]

    #drop events where area is zero
    hulls = [hulls[i] for i in range(len(hulls)) if area_origs[i]>0]    
    contours = [contours[i] for i in range(len(contours)) if area_origs[i]>0]
    mu_origs = [mu_origs[i] for i in range(len(mu_origs)) if area_origs[i]>0]
    mu_hulls = [mu_hulls[i] for i in range(len(mu_hulls)) if area_origs[i]>0]
    area_hulls = [area_hulls[i] for i in range(len(area_hulls)) if area_origs[i]>0]
    area_origs = [area_origs[i] for i in range(len(area_origs)) if area_origs[i]>0]
    
    
    pos_x = [int(mu_orig['m10']/mu_orig['m00']) for mu_orig in mu_origs]
    pos_y = [int(mu_orig['m01']/mu_orig['m00']) for mu_orig in mu_origs]

    
    if selectcell == "smooth":
        #compute the area ratio (roughness of contour)
        area_ratio = np.array(area_hulls)/np.array(area_origs)
        #get the contour with minimum roughness (smooth contour)
        sorter = np.argsort(area_ratio) #smallest first

    if selectcell == "centered":
        #select  contour that is closest to the center of the image. 
        #In iPAC, cells are usually in the center.
        mid_x,mid_y = mask.shape[0]/2,mask.shape[1]/2 #middle of the image
        BB = [cv2.boundingRect(c) for c in contours] #get a bounding box around the object
        distances = [np.sqrt((mid_x-bb[0])**2 + (mid_y-bb[1])**2) for bb in BB]
        sorter = np.argsort(distances) #smallest first
    
    #sort values with respect to chosen metric (area_ratio or distance)
    contours = [contours[s] for s in sorter]
    hulls = [hulls[s] for s in sorter]
    pos_x = [pos_x[s] for s in sorter]
    pos_y = [pos_y[s] for s in sorter]
    mu_origs = [mu_origs[s] for s in sorter]
    area_origs = [area_origs[s] for s in sorter]
    area_hulls = [area_hulls[s] for s in sorter]
    
    # draw mask of the chosen contour
    mask = np.zeros_like(mask)
    cv2.drawContours(mask,contours,0,1,cv2.FILLED)# produce a contour that is filled inside

    hull = hulls[0]#[0:n_contours]
    pos_x = pos_x[0]
    pos_y = pos_y[0]    
    mu_orig = mu_origs[0]#[0:n_contours]
    area_orig = area_origs[0]#[0:n_contours]
    area_hull = area_hulls[0]#[0:n_contours]
  
    if area_orig>0:
        area_ratio = area_hull/area_orig
    else:
        area_ratio = np.nan

    arc = cv2.arcLength(hull, True)    
    circularity = 2.0 * np.sqrt(np.pi * mu_orig["m00"]) / arc


    dic = {"mask":mask,"pos_x":pos_x,"pos_y":pos_y,"area_orig":area_orig,"area_hull":area_hull,\
           "area_ratio":area_ratio,"circularity":circularity}
    return dic

def read_cif_raw(reader,number_cells=np.inf,channels=[0,1,2],mask_available=True):
    assert type(number_cells)==int or number_cells==np.inf, f"Number_cells is {number_cells}, but should be integer (e.g. {int(number_cells)}), or np.inf"
    assert type(channels)==list, "channels should be a list, e.g. [0,5,8]"
    assert len(channels)<=3, f"Too many channels were chosen (you chose {channels}). Currently only up to 3 channels are possible"
    assert type(mask_available)==bool, f"mask_available should be True or False, but {mask_available} was provided."   
    images = []
    no_error = True
    cell_nr = 0
    LINE_CLEAR = '\x1b[2K'

    while no_error and len(images)<number_cells:#no_error:
        try:
            #print progress every 10 images
            if len(images)%10==0:
                if number_cells==np.inf:
                    done_pct = (len(images)/(series_count))*100
                else:
                    done_pct = (len(images)/(number_cells))*100
                print(LINE_CLEAR,f"Done by {np.round(done_pct,2)}%",end = "\r")
            # read the (fluorescence) image (uint16)
            image = reader.read(c=None, series=cell_nr,rescale=False)[:,:, channels] #indexing ([:,:, channels]) is faster than iterating over channels!
            images.append(np.atleast_3d(image))
            if mask_available:
                cell_nr += 2 #there are images in even indices and masks in odd indices
            else:
                cell_nr += 1 #there are fl images for all indices
                
        except Exception as e:
            print(e)
            no_error = False
    # images = np.stack(images,axis=0)
    # images = np.squeeze(images) #in case, its grayscale, it removes the excessive axis
    return images


def read_cif_scaled(reader,number_cells=np.inf,channels=[0,1,2],mask_available=True,mask_channel=None,contour_selection="centered",pixel_size=1,channel_scaling_factors=None):
    """
    Read the images from an Amnis ImageStream file (.cif)

    Parameters
    ----------
    reader: reader object (reader = bioformats.formatreader.get_image_reader("tmp", path=path_cif))
    number_cells: int or np.inf
        Maximum number of images to retrieve. If np.inf is chosen, all images will be obtained
    channels: list of integers
        integers indicate which channels to use
    mask_available = bool
        In some .cif files, every second series is the mask for the previous image.
    mask_channel = int or None
        Only possible when mask_available. Indicate which channel to use for computing mask features
        If None, no mask features will be computed
    Returns
    -------
    cif_data: dictionary
    """
    assert type(number_cells)==int or number_cells==np.inf, f"Number_cells is {number_cells}, but should be integer (e.g. {int(number_cells)}), or np.inf"
    assert type(channels)==list, "channels should be a list, e.g. [0,5,8]"
    assert len(channels)<=3, f"Too many channels were chosen (you chose {channels}). Currently only up to 3 channels are possible"
    if type(mask_channel)==int: # a channel number was defined for the mask channel
        assert mask_available, f"Contradicting parameters: mask channel was provided (ch:{mask_channel}), but mask_available is False!"
    assert type(mask_available)==bool, f"mask_available should be True or False, but {mask_available} was provided."   
    if channel_scaling_factors!=None:
        assert type(channel_scaling_factors)==list, "channel_scaling_factors needs to be a list of integers"
        assert len(channel_scaling_factors)==len(channels), "Please define one scaling factor for each channel!"
        
    t0  = time.perf_counter()
    images,masks,pos_x,pos_y = [],[],[],[]
    area_orig,area_hull,area_ratio,circularity = [],[],[],[]
    bright_avg,bright_sd = [],[]
    cell_nr = 0
    no_error = True
    series_count = reader.rdr.getSeriesCount()
    if mask_available:
        series_count = int(series_count/2) #half of them are masks
    
    LINE_CLEAR = '\x1b[2K'
    while no_error and len(images)<number_cells:#no_error:
        try:
            #print progress every 10 images
            if len(images)%10==0:
                if number_cells==np.inf:
                    done_pct = (len(images)/(series_count))*100
                else:
                    done_pct = (len(images)/(number_cells))*100
                print(LINE_CLEAR,f"Done by {np.round(done_pct,2)}%",end = "\r")

            # read the (fluorescence) image (uint16)
            image = reader.read(c=None, series=cell_nr,rescale=False)[:,:, channels] #indexing ([:,:, channels]) is faster than iterating over channels!
            
            #perform scaling of the channel intensity
            image = uint16_to_uint8(image,channel_scaling_factors) #convert to uint8 

            # Get mid-point of image for subsequent cropping
            mid_x = np.around(image.shape[0]/2.0)*pixel_size
            mid_y = np.around(image.shape[1]/2.0)*pixel_size
            # cropping
            image = image_crop_pad_cv2(images=[image],pos_x=[mid_x],pos_y=[mid_y],pix=pixel_size,final_h=image_size,final_w=image_size)[0]
            images.append(np.atleast_3d(image))

            if type(mask_channel)==int: # a mask channel was selected
                mask = reader.read(c=mask_channel, series=cell_nr+1,rescale=False)
                # crop mask equally like the image
                mask = image_crop_pad_cv2(images=[mask],pos_x=[mid_x],pos_y=[mid_y],pix=pixel_size,final_h=image_size,final_w=image_size)[0]

                contour_feats = get_contour_features(mask,selectcell=contour_selection)

                pos_x.append(contour_feats["pos_x"]*pixel_size)
                pos_y.append(contour_feats["pos_y"]*pixel_size)

                area_orig.append(contour_feats["area_orig"])
                area_hull.append(contour_feats["area_hull"])
                area_ratio.append(contour_feats["area_ratio"])
                circularity.append(contour_feats["circularity"])
                mask = contour_feats["mask"]
                masks.append(mask)

                output = get_brightness(image,mask)
                bright_avg.append(output["bright_avg"])
                bright_sd.append(output["bright_sd"])
            else:
                pos_x.append(mid_x)
                pos_y.append(mid_y)
                area_orig.append(np.nan)
                area_hull.append(np.nan)
                area_ratio.append(np.nan)
                circularity.append(np.nan)
                masks.append(np.nan)
                bright_sd.append(np.nan)
                bright_avg.append(np.nan)
            if mask_available:
                cell_nr += 2 #there are images in even indices and masks in odd indices
            else:
                cell_nr += 1 #there are fl images for all indices
                
        except Exception as e:
            print(e)
            no_error = False
    
    t1 = time.perf_counter()
    dt = np.round(t1-t0,2)
    
    images = np.stack(images,axis=0)
    images = np.squeeze(images) #in case, its grayscale, it removes the excessive axis
    if len(images.shape)==4 and images.shape[-1]==2: #there is a "channels"-dimension, indicating two channels -> add a third empty channel
        empty = np.zeros(shape=(images.shape[0],images.shape[1],images.shape[2],1),dtype=np.uint8)
        images = np.c_[images,empty]
    masks = np.stack(masks,axis=0)
    masks = np.squeeze(masks)

    print(f"Found {len(images)} images")
    print(f"Time needed to process cif file: {dt}s")
    
    cif_data = {"channel_scaling_factors":channel_scaling_factors,
                "images":images,"masks":masks, "pos_x":pos_x, "pos_y":pos_y,
                "area_orig":area_orig,"area_hull":area_hull,"area_ratio":area_ratio,
                "circularity":circularity,"bright_avg":bright_avg,
                "bright_sd":bright_sd,"pixel_size":pixel_size,"channel_scaling_factors":channel_scaling_factors}
    return cif_data

def write_rtdc(path_rtdc,cif_data):
    # initialize empty .rtdc file
    with h5py.File(path_rtdc,'w') as hdf:
        #many attributes are required (so it can run in ShapeOut2 - but only for Grayscale images anyway...)
        hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
        hdf.attrs["experiment:event count"] = cif_data["images"].shape[0]
        hdf.attrs["experiment:run index"] = 1
        hdf.attrs["experiment:sample"] = sample_name
        hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")
        hdf.attrs["imaging:flash device"] = "AMNIS"
        hdf.attrs["imaging:flash duration"] = 2.0
        hdf.attrs["imaging:frame rate"] = 2000
        hdf.attrs["imaging:pixel size"] = cif_data["pixel_size"]
        hdf.attrs["imaging:roi poition x"] = 629.0
        hdf.attrs["imaging:roi poition x"] = 512.0
        hdf.attrs["imaging:roi size x"] = 64
        hdf.attrs["imaging:roi size y"] = 64
        hdf.attrs["fluorescence:sample rate"] = 312500
        hdf.attrs["online_contour:bin area min"] = 10
        hdf.attrs["online_contour:bin kernel"] = 5
        hdf.attrs["online_contour:bin threshold"] = -6
        hdf.attrs["online_contour:image blur"] = 0
        hdf.attrs["online_contour:no absdiff"] = 1
        hdf.attrs["setup:channel width"] = 20
        hdf.attrs["setup:chip region"] = "channel"
        hdf.attrs["setup:flow rate"] = 0.06
        hdf.attrs["setup:flow rate sample"] = 0.02
        hdf.attrs["setup:flow rate sheath"] = 0.04
        hdf.attrs["setup:identifier"] = "AMNIS ImageStream"
        hdf.attrs["setup:medium"] = "CellCarrierB"
        hdf.attrs["setup:module composition"] = "AcCellerator"
        hdf.attrs["setup:software version"] = "dclab 0.9.0.post1 | dclab 0.9.1"
        hdf.attrs["fluorescence:channel_scaling_factors"] = cif_data["channel_scaling_factors"]
        
        # save images
        if len(cif_data["images"].shape) == 3: #plain single channel "grayscale" image
            maxshape_img = (None,*cif_data["images"].shape[1:])
            # maxshape_img = (None, cif_data["images"].shape[1], cif_data["images"].shape[2],cif_data["images"].shape[3])
            dset = hdf.create_dataset("events/image", data=cif_data["images"], dtype=np.uint8,maxshape=maxshape_img,fletcher32=True,chunks=True)
            dset.attrs.create('CLASS', np.string_('IMAGE'))
            dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
            dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
        if len(cif_data["images"].shape) == 4: # multichannel!
            maxshape_img_ch = (None,*cif_data["images"][:,:,:,0].shape[1:])
            print("Saving each channel as individual item in hdf...")
            for layer in range(cif_data["images"].shape[-1]): #iterate over three layers (RGB)
                if layer==0:
                    dset = hdf.create_dataset("events/image", data=cif_data["images"][:,:,:,layer], dtype=np.uint8,maxshape=maxshape_img_ch,fletcher32=True,chunks=True)
                else:
                    dset = hdf.create_dataset(f"events/image_ch{layer}", data=cif_data["images"][:,:,:,layer], dtype=np.uint8,maxshape=maxshape_img_ch,fletcher32=True,chunks=True)
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))

        # save position of cell
        hdf.create_dataset("events/pos_x", data=cif_data["pos_x"], dtype=np.float32,maxshape=(None,))
        hdf.create_dataset("events/pos_y", data=cif_data["pos_y"], dtype=np.float32,maxshape=(None,))


        # save masks if they are available:
        if "masks" in cif_data.keys():
            maxshape_mask = (None, cif_data["masks"].shape[1], cif_data["masks"].shape[2])
            dset = hdf.create_dataset("events/mask", data=cif_data["masks"], dtype=np.uint8,maxshape=maxshape_mask,fletcher32=True,chunks=True)
            dset.attrs.create('CLASS', np.string_('IMAGE'))
            dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
            dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_TRUECOLOR'))

        #save contour features if they are available
        if "area_orig" in cif_data.keys():
            area_um = np.array(cif_data["area_hull"])*pixel_size
            hdf.create_dataset("events/area_um", data=area_um, dtype=np.float32,maxshape=(None,))
            hdf.create_dataset("events/area_msd", data=cif_data["area_orig"], dtype=np.float32,maxshape=(None,))
            hdf.create_dataset("events/area_cvx", data=cif_data["area_hull"], dtype=np.float32,maxshape=(None,))
            hdf.create_dataset("events/area_ratio", data=cif_data["area_ratio"], dtype=np.float32,maxshape=(None,))
            hdf.create_dataset("events/circ", data=cif_data["circularity"], dtype=np.float32,maxshape=(None,))
        
        #save brightness features if they are available
        if "bright_avg" in cif_data.keys():
            hdf.create_dataset("events/bright_avg", data=cif_data["bright_avg"], dtype=np.float32,maxshape=(None,))
            hdf.create_dataset("events/bright_sd", data=cif_data["bright_sd"], dtype=np.float32,maxshape=(None,))


def intensity_range_info(images):
    print("Get channel intesity ranges")
    channel_nr,intensity_min,intensity_max = [],[],[]
    intensity_p75,intensity_p95,intensity_p99,intensity_p999,intensity_p9999,\
    intensity_p99999 = [],[],[],[],[],[]
    for channel in range(images[0].shape[-1]):
        images_ch = [img[:,:,channel] for img in images]
        images_ch = [img.flatten() for img in images_ch]
        images_ch = [x for xs in images_ch for x in xs]
        images_ch = np.array(images_ch)
        
        channel_nr.append(channel)
        intensity_min.append(np.min(images_ch))
        intensity_p75.append(np.percentile(images_ch,75))
        intensity_p95.append(np.percentile(images_ch,95))
        intensity_p99.append(np.percentile(images_ch,99))
        intensity_p999.append(np.percentile(images_ch,99.9))
        intensity_p9999.append(np.percentile(images_ch,99.99))
        intensity_p99999.append(np.percentile(images_ch,99.999))


        intensity_max.append(np.max(images_ch))

    intensity_info = pd.DataFrame()
    intensity_info["channel_nr"] = channel_nr
    intensity_info["intensity_min"] = intensity_min
    intensity_info["intensity_75%_percentile"] = intensity_p75
    intensity_info["intensity_95%_percentile"] = intensity_p95
    intensity_info["intensity_99%_percentile"] = intensity_p99
    intensity_info["intensity_99.9%_percentile"] = intensity_p999
    intensity_info["intensity_99.99%_percentile"] = intensity_p9999
    intensity_info["intensity_99.999%_percentile"] = intensity_p99999

    intensity_info["intensity_max"] = intensity_max
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(intensity_info)

    return intensity_info

def plot_height_width_scatter(cif_sample):
    heights = [img.shape[0] for img in cif_sample]
    widths = [img.shape[1] for img in cif_sample]
    # heights = np.unique(heights)
    # widths = np.unique(widths)
    plt.figure(1,figsize=(5,5))
    plt.scatter(widths,heights,c=None)
    plt.scatter(np.max(widths),np.max(heights),c="red")
    
    plt.xlabel("Image widths [pixels]")
    plt.ylabel("Image heights [pixels]")
    plt.title(f"Max height={np.max(heights)}, Max width={np.max(widths)}")

######################MAKE USE OF THE FUNCTIONS ABOVE##################

# print("Show all channel-images for one cell")
# show_cell_from_cif(0)

# ### get a few image to learn about the fluorescence intensities
# cif_sample = read_cif_raw(reader,number_cells=250,channels=channels,mask_available=True)

# ### scatterplot showing image heights and widths
# plot_height_width_scatter(cif_sample)
# ### -> it looks like most images are smaller than 80x80 pixels -> use that for cropping -> image_size = 80

# # # ### Get an overview of the intensity values for each channel
# intensity_info = intensity_range_info(cif_sample)

# ### check the table "intensity_info". You can e.g. use the 99.99th percentile for channel_fluo_upper_limit
# ### Make sure channel_fluo_upper_limit are some nice numbers that you can use in 
# ### all experiments
# # channel_fluo_upper_limit = [800,800,3000] # estimation of the max value of the fluorescence intensities
# channel_scaling_factors = [255/x for x in channel_fluo_upper_limit] # multiply by 255 since final images should be uint8 (0-255)
# print(f"Will use following channel_scaling_factors: {channel_scaling_factors}")

# ### Estimate the channel fluorescence intensity ranges based on first n cells
# cif_sample_scaled = read_cif_scaled(reader,number_cells=5,channels=channels,
#                     mask_available=True,mask_channel=None,
#                     contour_selection=None,pixel_size=pixel_size,
#                     channel_scaling_factors=channel_scaling_factors)

# ### plot these cells to have a look if they are nice enough
# show_cell_from_array(cif_sample_scaled)

### All parameters are defined! Now it's time to read all data
print("Read all images from cif")
cif_data_scaled = read_cif_scaled(reader,number_cells=number_cells,channels=channels,
                    mask_available=mask_available,mask_channel=mask_channel,
                    contour_selection=contour_selection,pixel_size=pixel_size,
                    channel_scaling_factors=channel_scaling_factors)

print("Writing to .rtdc")
write_rtdc(path_rtdc,cif_data_scaled)

# stopping the java virtual machine
javabridge.kill_vm()


















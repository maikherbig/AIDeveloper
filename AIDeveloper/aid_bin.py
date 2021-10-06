# -*- coding: utf-8 -*-
"""
aid_bin
some useful functions that I want to keep separate to
keep the backbone script shorter
---------
@author: maikherbig
"""

import os,shutil,json,re,urllib
import numpy as np
import dclab
import h5py,time,datetime
import six,tarfile, zipfile
import hashlib
import warnings
import pathlib
import aid_img

import aid_start #import a module that sits in the AIDeveloper folder
dir_root = os.path.dirname(aid_start.__file__)#ask the module for its origin

def save_aid_settings(Default_dict):
    dir_settings = os.path.join(dir_root,"aid_settings.json")#dir to settings
    #Save the layout to Default_dict
    with open(dir_settings, 'w') as f:
        json.dump(Default_dict,f)

def splitall(path):
    """
    Credit goes to Trent Mick
    SOURCE:
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def hashfile(fname, blocksize=65536, count=0, constructor=hashlib.md5,
             hasher_class=None):
    """Compute md5 hex-hash of a file
    Parameters
    ----------
    fname: str or pathlib.Path
        path to the file
    blocksize: int
        block size in bytes read from the file
        (set to `0` to hash the entire file)
    count: int
        number of blocks read from the file
    hasher_class: callable
        deprecated, see use `constructor` instead
    constructor: callable
        hash algorithm constructor
    """
    if hasher_class is not None:
        warnings.warn("The `hasher_class` argument is deprecated, please use "
                      "`constructor` instead.")
        constructor = hasher_class
    hasher = constructor()
    fname = pathlib.Path(fname)
    with fname.open('rb') as fd:
        buf = fd.read(blocksize)
        ii = 0
        while len(buf) > 0:
            hasher.update(buf)
            buf = fd.read(blocksize)
            ii += 1
            if count and ii == count:
                break
    return hasher.hexdigest()

def obj2bytes(obj):
    """Bytes representation of an object for hashing"""
    if isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, pathlib.Path):
        return obj2bytes(str(obj))
    elif isinstance(obj, (bool, int, float)):
        return str(obj).encode("utf-8")
    elif obj is None:
        return b"none"
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()
    elif isinstance(obj, tuple):
        return obj2bytes(list(obj))
    elif isinstance(obj, list):
        return b"".join(obj2bytes(o) for o in obj)
    elif isinstance(obj, dict):
        return obj2bytes(sorted(obj.items()))
    elif hasattr(obj, "identifier"):
        return obj2bytes(obj.identifier)
    elif isinstance(obj, h5py.Dataset):
        return obj2bytes(obj[0])
    else:
        raise ValueError("No rule to convert object '{}' to string.".
                         format(obj.__class__))

def hashfunction(rtdc_path):
    """Hash value based on file name and content"""
    tohash = [os.path.basename(rtdc_path),
              # Hash a maximum of ~1MB of the hdf5 file
              hashfile(rtdc_path, blocksize=65536, count=20)]
    return hashlib.md5(obj2bytes(tohash)).hexdigest()


def load_rtdc(rtdc_path):
    """
    This function load .rtdc files using dclab and takes care of catching all
    errors
    """
    try:
        try:
            #sometimes there occurs an error when opening hdf files,
            #therefore try opening a second time in case of an error.
            #This is very strange, and seems like a dirty solution,
            #but I never saw it failing two times in a row
            rtdc_ds = h5py.File(rtdc_path, 'r')
        except:
            rtdc_ds = h5py.File(rtdc_path, 'r')
        return False,rtdc_ds #failed=False
    except Exception as e:
        #There is an issue loading the files!
        return True,e

def print_ram_example(n):
    #100k cropped images (64x64 pix unsigned integer8) need 400MB of RAM:
    Imgs = []
    sizex,sizey = 64,64
    for i in range(100000):
        Imgs.append(np.random.randint(low=0,high=255,size=(sizex,sizey)))
    Imgs = np.array(Imgs)
    Imgs = Imgs.astype(np.uint8)
    print(str(Imgs.shape[0]) + " images (uint8) of size " + str(sizex) + "x" + str(sizey) + " pixels take " +str(Imgs.nbytes/1048576.0) +" MB of RAM")

def calc_ram_need(crop):
    crop = int(crop)
    n=1000
    #100k cropped images (64x64 pix unsigned integer8) need 400MB of RAM:
    Imgs = []
    sizex,sizey = crop,crop
    for i in range(n):
        Imgs.append(np.random.randint(low=0,high=255,size=(sizex,sizey)))
    Imgs = np.array(Imgs)
    Imgs = Imgs.astype(np.uint8)
    MB = Imgs.nbytes/1048576.0 #Amount of RAM for 1000 images
    MB = MB/float(n)
    return MB

def metrics_using_threshold(scores,y_valid,threshold,target_index,thresh_on=True):
    nr_target_init = float(len(np.where(y_valid==target_index)[0])) #number of target cells in the initial sample

    conc_init = 100*nr_target_init/float(len(y_valid)) #concentration of the target cells in the initial sample
    scores_in_function = np.copy(scores)
    
    if thresh_on==True:    
        #First: check the scores_in_function of the sorting index and adjust them using the threshold
        pred_thresh = np.array([1 if p>threshold else 0 for p in scores_in_function[:,target_index]])
        #replace the corresponding column in the scores_in_function
        scores_in_function[:,target_index] = pred_thresh

    #Finally use argmax for the rest of the predictions (threshold can only be applied to one index)
    pred = np.argmax(scores_in_function,axis=1)

    ind = np.where( pred==target_index )[0] #which cells are predicted to be target cells?
    y_train_prime = y_valid[ind] #get the correct label of those cells
    where_correct = np.where( y_train_prime==target_index )[0] #where is this label equal to the target label
    nr_correct = float(len(where_correct)) #how often was the correct label in the target

    if len(y_train_prime)==0:
        conc_target_cell=0
    else:
        conc_target_cell = 100.0*(nr_correct/float(len(y_train_prime))) #divide nr of correct target cells by nr of total target cells
    if conc_init==0:
        enrichment=0
    else:
        enrichment = conc_target_cell/conc_init

    if nr_target_init==0:
        yield_ = 0
    else:
        yield_ = (nr_correct/nr_target_init)*100.0
    dic = {"scores":scores,"pred":pred,"conc_target_cell":conc_target_cell,"enrichment":enrichment,"yield_":yield_}
    return dic

def find_files(user_selected_path,paths,hashes):
    assert len(paths) == len(hashes)
    #Create a list of files in that folder
    Paths_available,Fnames_available = [],[]
    for root, dirs, files in os.walk(user_selected_path):
        for file in files:
            if file.endswith(".rtdc"):
                Paths_available.append(os.path.join(root, file))
                Fnames_available.append(file)
    #Iterate through the list of given paths and search each measurement in Files
    Paths_new,Info = [],[]
    for i in range(len(paths)):
        path_ = paths[i]
        hash_ = hashes[i]
        p,fname = os.path.split(path_)
        #where does a file of that name exist:
        ind = [fname_new == fname for fname_new in Fnames_available] #there could be several since they might originate from the same measurement, but differently filtered
        paths_new = list(np.array(Paths_available)[ind]) #get the corresponding paths to the files
        #Therfore, open the files and get the hashes
            #hash_new = [(dclab.rtdc_dataset.RTDC_HDF5(p)).hash for p in paths_new] #since reading hdf sometimes does cause error, better try and repaeat if necessary
        hash_new = []
        for p in paths_new:
            failed,rtdc_ds = load_rtdc(p)
            if failed:
                print("Error occurred during loading file\n"+str(p)+"\n"+str(rtdc_ds))
            else:
                hash_new.append(hashfunction(p))

        ind = [ h==hash_ for h in hash_new ] #where do the hashes agree?
        #get the corresponding Path_new
        path_new = list(np.array(paths_new)[ind])
        if len(path_new)==1:
            Paths_new.append(str(path_new[0]))
            Info.append("Found the required file!")
        elif len(path_new)==0:
            Paths_new.append([])
            Info.append("File missing!")
        if len(path_new)>1:
            Paths_new.append(str(path_new[0]))
            Info.append("Found the required file multiple times! Choose one!")
    return(Paths_new,Info)


#: Chunk size for storing HDF5 data
CHUNK_SIZE = 100

def store_contour(h5group,name, data, compression):
    if not isinstance(data, (list, tuple)):
        # single event
        data = [data]
    grp = h5group.require_group(name)
    curid = len(grp.keys())
    for ii, cc in enumerate(data):
        grp.create_dataset("{}".format(curid + ii),
                           data=cc,
                           fletcher32=True,
                           compression=compression)


def store_image(h5group, name, data, compression, background=False):
    """Store image data in an HDF5 group
    Parameters
    ----------
    h5group: h5py.Group
        The group (usually "events") where to store the image data
    data: 2d or 3d ndarray
        The image data. If 3d, then the first axis enumerates
        the images.
    compression: str
        Dataset compression method
    background: bool
        If set to False (default), then the regular "image" is stored;
        If set to True, then the background image ("image_bg") is
        stored.
    """
    if len(data.shape) == 2:
        # single event
        data = data.reshape(1, data.shape[0], data.shape[1])
    if name not in h5group:
        maxshape = (None, data.shape[1], data.shape[2])
        chunks = (CHUNK_SIZE, data.shape[1], data.shape[2])
        dset = h5group.create_dataset(name,
                                      data=data,
                                      dtype=np.uint8,
                                      maxshape=maxshape,
                                      chunks=chunks,
                                      fletcher32=True,
                                      compression=compression)
        # Create and Set image attributes:
        # HDFView recognizes this as a series of images.
        # Use np.string_ as per
        # http://docs.h5py.org/en/stable/strings.html#compatibility
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
    else:
        dset = h5group[name]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_mask(h5group,name, data, compression):
    # store binary mask data as uint8 to allow visualization in HDFView
    data = np.asarray(data, dtype=np.uint8)
    if data.max() != 255 and data.max() != 0 and data.min() == 0:
        data = data / data.max() * 255
    if len(data.shape) == 2:
        # single event
        data = data.reshape(1, data.shape[0], data.shape[1])
    if "mask" not in h5group:
        maxshape = (None, data.shape[1], data.shape[2])
        chunks = (CHUNK_SIZE, data.shape[1], data.shape[2])
        dset = h5group.create_dataset(name,
                                      data=data,
                                      dtype=np.uint8,
                                      maxshape=maxshape,
                                      chunks=chunks,
                                      fletcher32=True,
                                      compression=compression)
        # Create and Set image attributes
        # HDFView recognizes this as a series of images
        dset.attrs.create('CLASS', np.string_('IMAGE'))
        dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
        dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
    else:
        dset = h5group[name]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_scalar(h5group, name, data, compression):
    if np.isscalar(data):
        # single event
        data = np.atleast_1d(data)
    if name not in h5group:
        h5group.create_dataset(name,
                               data=data,
                               maxshape=(None,),
                               chunks=(CHUNK_SIZE,),
                               fletcher32=True,
                               compression=compression
                               )
    else:
        dset = h5group[name]
        oldsize = dset.shape[0]
        dset.resize(oldsize + data.shape[0], axis=0)
        dset[oldsize:] = data


def store_trace(h5group,name, data, compression):
    firstkey = sorted(list(data.keys()))[0]
    if len(data[firstkey].shape) == 1:
        # single event
        for dd in data:
            data[dd] = data[dd].reshape(1, -1)
    # create trace group
    grp = h5group.require_group(name)

    for flt in data:
        # create traces datasets
        if flt not in grp:
            maxshape = (None, data[flt].shape[1])
            chunks = (CHUNK_SIZE, data[flt].shape[1])
            grp.create_dataset(flt,
                               data=data[flt],
                               maxshape=maxshape,
                               chunks=chunks,
                               fletcher32=True,
                               compression=compression)
        else:
            dset = grp[flt]
            oldsize = dset.shape[0]
            dset.resize(oldsize + data[flt].shape[0], axis=0)
            dset[oldsize:] = data[flt]


def write_rtdc(fname,rtdc_datasets,X_valid,Indices,cropped=True,color_mode='Grayscale',xtra_in=[]):
    """
    fname - path+filename of file to be created
    rtdc_datasets - list paths to rtdc data-data-sets
    X_valid - list containing numpy arrays. Each array contains cropped images of individual measurements corresponding to each rtdc_ds
    Indices - list containing numpy arrays. Each array contais index values which refer to the index of the cell in the original rtdc_ds
    """
    #Check if a file with name fname already exists:
    if os.path.isfile(fname):
        os.remove(fname) #delete it
        print("overwrite existing file")
    
    index_new = np.array(range(1,int(np.sum(np.array([len(I) for I in Indices]))+1))) #New index. Will replace the existing index in order to support viewing imges in shapeout   

    #quickly check the image-format of the validation data:
    images_shape = []
    for i in range(len(rtdc_datasets)):
        failed,rtdc_ds = load_rtdc(rtdc_datasets[i])
        if failed:
            print("Error occurred during loading file\n"+str(rtdc_datasets[i])+"\n"+str(rtdc_ds))
        else:
            #get the shape for all used files
            images_shape.append(rtdc_ds["events"]["image"].shape) 
    #Allow RGB export only, of all files have 3 channels
    if cropped==False: #only if the original images should be exported
        #the length of the images is shape=3 for grayscale images and =4 for RGB images
        images_shape_l = [len(images_shape[i]) for i in range(len(images_shape))]
        channels = [s[-1] for s in images_shape] #get the nr of channels in each file
        if images_shape_l==len(images_shape_l)*[4]: #if all files have len(shape)=4
            if channels==len(channels)*[3]: #check if all files have 3 channels 
                print("All files have three channels. I will save the entire RGB information")
                color_mode = "RGB"
            else: #(some) files neither have 1 nor 3 channels. Not supported
                print("AID currently only supports single channel (grayscale) and three channel (RGB) data. Your data has following channels: "+str(channels))
                return
                #TODO: Maybe support more channels
        else: #not all files have three channels->ergo there is no way to keep the color-information:
            color_mode = "Grayscale"

    #########Only for RGB Images!: Collect all data first and write############ 
    if color_mode=='RGB': #or images_shape_max==4:
    #if len(X_valid)>0 and len(np.array(X_valid).shape)==5: #RGB image
        #Get all images,pos_x,pos_y:
        images,pos_x,pos_y = [],[],[]
        for i in range(len(rtdc_datasets)):
            failed,rtdc_ds = load_rtdc(rtdc_datasets[i])
            if failed:
                print("Error occurred during loading file\n"+str(rtdc_datasets[i])+"\n"+str(rtdc_ds))
            else:
                indices = Indices[i]
                if len(indices>0):
                    if cropped==False:
                        images.append([rtdc_ds["events"]["image"][ii] for ii in indices])
                    pos_x.append([rtdc_ds["events"]["pos_x"][ii] for ii in indices])
                    pos_y.append([rtdc_ds["events"]["pos_y"][ii] for ii in indices])
                    
        if cropped==True:
            images = X_valid
            #If the user want to export cropped images, then they will be
            #in the color mode of the model

        images = np.concatenate(images)
        pos_x = np.concatenate(pos_x)
        pos_y = np.concatenate(pos_y)
        
        #copy the empty Empty.rtdc
        shutil.copy(os.path.join(dir_root,"Empty.rtdc"),fname)
        
        maxshape = (None, images.shape[1], images.shape[2], images.shape[3])
        #Create rtdc_dataset
        hdf = h5py.File(fname,'a')
        hdf.create_dataset("events/image", data=images, dtype=np.uint8,maxshape=maxshape,fletcher32=True,chunks=True)
        hdf.create_dataset("events/pos_x", data=pos_x, dtype=np.int32)
        hdf.create_dataset("events/pos_y", data=pos_y, dtype=np.int32)
        hdf.create_dataset("events/index", data=index_new, dtype=np.int32)
        if len((np.array(xtra_in)).ravel())>0: #
            #hdf.create_dataset("xtra_in", data=xtra_in, dtype=np.float32)
            hdf.create_dataset('xtra_in', data=np.concatenate(xtra_in), compression="gzip", chunks=True, maxshape=(None,))
        
        #Adjust metadata:
        #"experiment:event count" = Nr. of images
        hdf.attrs["experiment:event count"]=images.shape[0]
        hdf.attrs["experiment:sample"]=fname
        hdf.attrs["imaging:pixel size"]=rtdc_ds.attrs["imaging:pixel size"]
        hdf.close()
        return

    Features,Trace_lengths,Mask_dims_x,Mask_dims_y,Img_dims_x,Img_dims_y = [],[],[],[],[],[]
    for i in range(len(rtdc_datasets)):
        failed,rtdc_ds = load_rtdc(rtdc_datasets[i])
        if failed:
            print("Error occurred during loading file\n"+str(rtdc_datasets[i])+"\n"+str(rtdc_ds))
        else:
            features = list(rtdc_ds["events"].keys())#rtdc_ds._events.keys()#all features
            Features.append(features)
    
            #The lengths of the fluorescence traces have to be equal, otherwise those traces also have to be dropped
            if "trace" in features:
                trace_lengths = [(rtdc_ds["events"]["trace"][tr][0]).size for tr in rtdc_ds["trace"].keys()]
                Trace_lengths.append(trace_lengths)
            #Mask Image dimensions have to be equal, otherwise those mask have to be dropped
            if "mask" in features:
                mask_dim = (rtdc_ds["events"]["mask"][0]).shape
                Mask_dims_x.append(mask_dim[0])
                Mask_dims_y.append(mask_dim[1])
            #Mask Image dimensions have to be equal, otherwise those mask have to be dropped
            if "image" in features:
                img_dim = (rtdc_ds["events"]["image"][0]).shape
                Img_dims_x.append(img_dim[0])
                Img_dims_y.append(img_dim[1])
 
    #Find common features in all .rtdc sets:
    def commonElements(arr): 
        # initialize result with first array as a set 
        result = set(arr[0]) 
        for currSet in arr[1:]: 
            result.intersection_update(currSet) 
        return list(result)     
    features = commonElements(Features)

    # features = ["index_orig"] + features
    # if "index" not in features:
    #     features = ["index"] + features
        
    if "trace" in features:
        Trace_lengths = np.concatenate(Trace_lengths)
        trace_lengths = np.unique(np.array(Trace_lengths))            
        if len(trace_lengths)>1:
            ind = np.where(np.array(features)!="trace")[0]
            features = list(np.array(features)[ind])
            print("Dropped traces becasue of unequal lengths")

    if "mask" in features:
        mask_dim_x = np.unique(np.array(Mask_dims_x))            
        mask_dim_y = np.unique(np.array(Mask_dims_y))            
        if len(mask_dim_x)>1 or len(mask_dim_y)>1:
            ind = np.where(np.array(features)!="mask")[0]
            features = list(np.array(features)[ind])
            print("Dropped mask becasue of unequal image sizes")

    if "image" in features:
        img_dim_x = np.unique(np.array(Img_dims_x))            
        img_dim_y = np.unique(np.array(Img_dims_y))            
        if len(img_dim_x)>1 or len(img_dim_y)>1:
            print("Unequal image dimensions -> Force export of cropped images!")
            cropped = True

    for i in range(len(rtdc_datasets)):
        failed,rtdc_ds = load_rtdc(rtdc_datasets[i])
        if failed:
            print("Error occurred during loading file\n"+str(rtdc_datasets[i])+"\n"+str(rtdc_ds))
        else:
            indices = Indices[i]
            Images = X_valid[i]
            
            if len(Images)>0 and len(Images.shape)==3:
                index_new_ = index_new[0:len(indices)]
                index_new = np.delete(index_new,range(len(indices)))

                h5obj = h5py.File(fname,'a')
                events = h5obj.require_group("events")
                
                # with dclab.rtdc_dataset.write_hdf5.write(path_or_h5file=fname,meta=meta, mode="append") as h5obj:
                # write each feature individually
                for feat in features:
                    if feat == "contour":
                        print("Omitting contours")
                        #cont_list = [rtdc_ds["events"]["contour"][ii] for ii in indices]
                        #store_contour(h5group=events,name=feat,data=cont_list, compression="gzip")

                    # elif feat == "index_orig":
                    #     store_scalar(h5group=events,name=feat, data=index_new_, compression="gzip")

                    # elif feat == "index_orig":
                    #     values = np.array(rtdc_ds["events"]["index"])[indices]
                    #     store_scalar(h5group=events,name=feat, data=values, compression="gzip")
                    
                    elif feat=="mask":# in ["mask", "image"]:
                        mask = np.array(rtdc_ds["events"]["mask"])[indices]
                        if cropped:
                            pix = rtdc_ds.attrs["imaging:pixel size"]
                            pos_x = np.array(rtdc_ds["events"]["pos_x"])[indices]/pix
                            pos_y = np.array(rtdc_ds["events"]["pos_y"])[indices]/pix
                            img_dim_x = Images[0].shape[1]
                            img_dim_y = Images[0].shape[0]
                            mask = aid_img.image_crop_pad_cv2(list(mask),pos_x,pos_y,pix,img_dim_x,img_dim_y,padding_mode="cv2.BORDER_CONSTANT")
                        
                        store_mask(h5group=events,name=feat, data=mask, compression="gzip")

                    elif "image" in feat:
                        if cropped and feat=="image":
                            image_list = np.array(Images)
                        elif cropped and feat != "image":#there are images of another channel, they need ot be cropped first
                            pix = rtdc_ds.attrs["imaging:pixel size"]
                            pos_x = np.array(rtdc_ds["events"]["pos_x"])[indices]/pix
                            pos_y = np.array(rtdc_ds["events"]["pos_y"])[indices]/pix
                            img_dim_x = Images[0].shape[1]
                            img_dim_y = Images[0].shape[0]
                            image_list = np.array(rtdc_ds["events"][feat])[indices]
                            image_list = aid_img.image_crop_pad_cv2(list(image_list),pos_x,pos_y,pix,img_dim_x,img_dim_y,padding_mode="cv2.BORDER_CONSTANT")
                            image_list = np.array(image_list)
                        else:
                            image_list = np.array(rtdc_ds["events"][feat])[indices]
                        store_image(h5group=events,name=feat, data=image_list, compression="gzip")

                    elif feat == "trace":
                        # create events group
                        trace = np.array(rtdc_ds["events"]["trace"])[indices]
                        dclab.rtdc_dataset.write_hdf5.store_trace(h5group=events,name=feat,data=trace,compression="gzip")

                    elif feat == "pos_x" and cropped==True:
                        values = np.zeros(shape=len(indices))+np.round(img_dim_x/2.0)*rtdc_ds.attrs["imaging:pixel size"]
                        dclab.rtdc_dataset.write_hdf5.store_scalar(h5group=events, 
                            name=feat, data=values, compression="gzip")

                    elif feat == "pos_y" and cropped==True:
                        values = np.zeros(shape=len(indices))+np.round(img_dim_y/2.0)*rtdc_ds.attrs["imaging:pixel size"]
                        dclab.rtdc_dataset.write_hdf5.store_scalar(h5group=events,
                            name=feat, data=values, compression="gzip")
                        
                    else:
                        values = np.array(rtdc_ds["events"][feat])[indices]
                        dclab.rtdc_dataset.write_hdf5.store_scalar(h5group=events, 
                            name=feat, data=values, compression="gzip")

                #Adjust metadata:
                #"experiment:event count" = Nr. of images
                h5obj.attrs["experiment:event count"] = np.sum(np.array([len(indi) for indi in Indices])) 
                if cropped:
                    #Adjust the meta for cropped images
                    img_dim_x = Images[0].shape[1]
                    img_dim_y = Images[0].shape[0]
                    h5obj.attrs["imaging:roi size x"] = img_dim_x
                    h5obj.attrs["imaging:roi size y"] = img_dim_y

                meta_keys = list(rtdc_ds.attrs)
                if "experiment:date" not in meta_keys:
                    h5obj.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
                if "experiment:time" not in meta_keys:
                    h5obj.attrs["experiment:time"] = time.strftime("%Y-%m-%d")
                    
                for meta_key in meta_keys:
                    h5obj.attrs[meta_key] = rtdc_ds.attrs[meta_key]

                h5obj.close()
                    
    #Append xtra_in data to the rtdc file
    if len((np.array(xtra_in)).ravel())>0: #in case there is some xtra_in data
        with h5py.File(fname, 'a') as rtdc_h5:
            try:
                rtdc_h5.create_dataset('xtra_in', data=np.concatenate(xtra_in), compression="gzip", chunks=True, maxshape=(None,))
                rtdc_h5.close()
            except:
                pass


def create_temp_folder():
    temp_path = os.path.join(dir_root,"temp")
    if os.path.exists(temp_path):
        print("Found existing temporary folder: "+temp_path)
#        print("Delete all contents of that folder")
#        shutil.rmtree(temp_path,ignore_errors=True)
#        time.sleep(0.5)
#        try:
#            os.mkdir(temp_path)
#        except:
#            print("Could not delete temp folder. Files probably still in use!")
#            pass

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        print("Created temporary folder: "+temp_path)
    return temp_path


#def delete_temp_folder():
#    temp_path = os.path.join(dir_root,"temp")
#    if os.path.exists(temp_path):
#        print("Found existing temporary folder: "+temp_path)
#        print("Delete all contents of that folder")
#        shutil.rmtree(temp_path,ignore_errors=True)
#        time.sleep(0.5)
#        try:
#            os.mkdir(temp_path)
#        except:
#            print("Could not delete temp folder. Files probably still in use!")
#            pass
#
#    if not os.path.exists(temp_path):#temp folder does not exist
#        os.mkdir(temp_path)#create the folder
#        print("Created temporary folder: "+temp_path)
#    return temp_path

def count_temp_folder():
    """
    Count the number of folders within the temporary folder
    """
    temp_path = os.path.join(dir_root,"temp")
    if os.path.exists(temp_path):#if temp even exists...
        files = os.listdir(temp_path)
        nr_temp_files = len(files)
    else:
        #there is no temp folder!
        create_temp_folder()#create a temp folder
        nr_temp_files = 0
    return nr_temp_files 

def open_temp():
    temp_path = create_temp_folder()
    os.startfile(temp_path)


def ram_compare_data(ram_dic,new_dic):    
    #compare the rtdc filenames:
    new_rtdc_paths = [a["rtdc_path"] for a in new_dic["SelectedFiles"]]
    ram_rtdc_paths = list(ram_dic["rtdc_path"])
    test_rtdc_paths = set(ram_rtdc_paths)==set(new_rtdc_paths)

    #Compare the image shape (size)
    ram_imgshape = ram_dic["Cropped_Images"][0].shape
    ram_imgcrop = ram_imgshape[1]
    new_imgcrop = new_dic["cropsize2"]
    test_imgcrop = ram_imgcrop==new_imgcrop
    
    #Compare the colormode
    if len(ram_imgshape)==3:
        ram_colormode = "grayscale"
    elif len(ram_imgshape)==4 and ram_imgshape[-1]==3:
        ram_colormode = "rgb"
    else:
        print("Image dimension not supported")
    new_colormode = new_dic["color_mode"].lower()
    test_colormode = ram_colormode==new_colormode
      
    #compare the number of images
    ram_nr_images = [a.shape[0] for a in ram_dic["Cropped_Images"]]
    new_nr_images = [a["nr_images"] for a in new_dic["SelectedFiles"]]
    test_nr_images = set(ram_nr_images)==set(new_nr_images)
 
    dic = {"test_rtdc_paths":test_rtdc_paths,"test_imgcrop":test_imgcrop,"test_colormode":test_colormode,"test_nr_images":test_nr_images}
    #Are all tests poisitve (True)?
    alltrue = all(dic.values())
    return alltrue
    
def download_zip(url_zip,fpath):
    """
    Download a zip file
    Parameters
    ----------
    url: str, URL to the downloadable zip file (e.g. https://github.com/maikherbig/AIDeveloper/releases/download/1.0.1-update/AIDeveloper_1.0.1-update.zip)
    fpath: str, path to store the downloaded data locally
    """
    error_msg = 'URL fetch failure on {}: {} -- {}'
        
    try:
      try:
        six.moves.urllib.request.urlretrieve(url=url_zip,filename=fpath)
      except six.moves.urllib.error.HTTPError as e:
        raise Exception(error_msg.format(url_zip, e.code, e.msg))
      except six.moves.urllib.error.URLError as e:
        raise Exception(error_msg.format(url_zip, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as a:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise

def extract_archive(file_path, path='.', archive_format='auto'):
  """
  from:
      https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/data_utils.py#L168-L297
      
  Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
  Arguments:
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, six.string_types):
    archive_format = [archive_format]

  #file_path = path_to_string(file_path)
  #path = path_to_string(path)

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False

def updates_ondevice():
    """
    Searches the directory of AIDeveloper for available updates (zip folders)
    Returns
    list: list contains strings, each is a tag of an -update version    
    """
    files = os.listdir(dir_root)
    files = [file for file in files if file.startswith("AIDeveloper_") and file.endswith(".zip")]
    update_zips = [file for file in files if "update" in file]
    backup_zips = [file for file in files if "backup" in file]
    ondevice_zips = update_zips+backup_zips
    tags_update = [file.split("AIDeveloper_")[1] for file in ondevice_zips]
    tags_update = [file.split(".zip")[0] for file in tags_update]
    return tags_update

def check_for_updates(this_version):
    """
    check GitHub for new releases (updates of AIDeveloper)
    check local (on device) AIDeveloper folder for "-update.zip" and "-backup.zip" files
    
    Parameters
    ----------
    this_version: str, version of the installed release
    """
    #pre-define some variables
    Errors = None
    latest_release = None
    url = ""
    changelog = ""
    tags_update_online = []
    tags_update_ondevice = []

    #check local (on device)
    tags_update_ondevice = updates_ondevice()
    
    #check online
    try:
        url_releases = "https://github.com/maikherbig/AIDeveloper/releases"
        content = urllib.request.urlopen(url_releases).read().decode('UTF-8')
        tags = content.split("/maikherbig/AIDeveloper/releases/tag/")[1:]
        tags = [t.split('">AIDeveloper')[0] for t in tags]

        #handle  -update versions separately
        tags_update_online = [a for a in tags if "update" in a]
        tags_update_online = [a.split("-update")[0]+"-update" for a in tags_update_online]+["Bleeding edge"]
        tags = [a for a in tags if not "update" in a]

        latest_release = tags[0]
        highest_dev = [latest_release+"_dev" in tag for tag in tags]
        ind = np.where(np.array(highest_dev)==True)[0]
        if len(ind)>0:
            highest_devs_tag = list(np.array(tags)[ind])
            #corresponding numbers
            devs_tag_values =  [list(map(int, re.findall(r'\d+', t))) for t in highest_devs_tag]
            highest_dev_ind = np.argmax([t[3] for t in devs_tag_values])
            #name of the highest dev
            latest_release = highest_devs_tag[highest_dev_ind]
            url = "https://github.com/maikherbig/AIDeveloper/releases/tag/"+latest_release
        else:
            url = "https://github.com/maikherbig/AIDeveloper/releases/tag/"+latest_release
        
        #if "latest_release" is different from "this_version" then, an update is available!
        if this_version==latest_release:
            #no need to update, overwrite latest_release variable
            latest_release = "You are up to date"
            changelog = "You are up to date"
        else:
            #Open the url of the latest release and get the changelog
            content = urllib.request.urlopen(url).read().decode('UTF-8')
            changelog = content.split('<div class="markdown-body">')[1].split("</p>\n  </div>")[0]
            changelog = "Changelog:\n"+changelog.lstrip()
        
    except Exception as e:
        #There is an issue. Maybe no internet connection...
        Errors = e
        
    dic = {"Errors":Errors,"latest_release":latest_release,"latest_release_url":url
           ,"changelog":changelog,"tags_update_online":tags_update_online,
           "tags_update_ondevice":tags_update_ondevice}

    return dic

def aideveloper_filelist():
    files = [
    "AIDeveloper.py",
    "aid_backbone.py",
    "aid_bin.py",
    "aid_dependencies_linux.txt",
    "aid_dependencies_mac.txt",
    "aid_dependencies_win.txt",
    "aid_dl.py",
    "aid_frontend.py",
    "aid_img.py",
    "aid_imports.py",
    "aid_start.py",
    "aid_settings.json",
    "Empty.rtdc",
    "layout_dark_notooltip.txt",
    "layout_dark.txt",
    "layout_darkorange_notooltip.txt",
    "layout_darkorange.txt",
    "main_icon_simple_04_48.ico",
    "main_icon_simple_04_256.icns",
    "main_icon_simple_04_256.ico",
    "model_zoo.py",
    "partial_trainability.py"]
    return files

def check_aid_scripts_complete(aid_directory):
    files = aideveloper_filelist()
    #complete path to files/folders
    files_paths = [os.path.join(aid_directory,file) for file in files] #full path

    #check that all required files are there    
    check = [os.path.isfile(path) or os.path.isdir(path) for path in files_paths]
    ind = [i for i, x in enumerate(check) if not x]       
    assert all(check), "Cannot create a backup! Following file(s) are missing: "+str([files_paths[i] for i in ind])

    return files

    
def backup_current_version(VERSION):
    """
    Collect following files and folder:
    """
    files = check_aid_scripts_complete(dir_root)

    #create a name for the zipfile (without overwriting) 
    path_save = "AIDeveloper_"+VERSION+"-backup.zip"
    path_save = os.path.join(dir_root,path_save)#path to save the update zip        
    if not os.path.exists(path_save):#if such a file does not yet exist...
        path_save = path_save
    else:#such a file already exists!
        #Avoid to overwriting existing file:
        print("Adding additional number since file exists!")
        i = 1
        while os.path.exists(path_save):
            path_save = "AIDeveloper_"+VERSION+"-backup_"+str(i)+".zip"
            path_save = os.path.join(dir_root,path_save)#path to save the update zip        
            i+=1

    #create a ZipFile object
    with zipfile.ZipFile(path_save, 'w',compression=zipfile.ZIP_DEFLATED) as zipObj:
        for file in files:
            path_orig = os.path.join(dir_root,file)
            zipObj.write(path_orig, os.path.basename(path_orig))

        dirName = os.path.join(dir_root,"art")
        #Iterate over all the files in "art" folder
        for folder, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                filePath = os.path.join(folder, filename)
                path_in_zip = os.path.join("art",filePath.split(os.sep+"art"+os.sep)[1])
                zipObj.write(filePath,path_in_zip)
               
    return path_save #return the filename of the backup file

def delete_current_version(delete_art=False):
    files = check_aid_scripts_complete(dir_root)
    files_paths = [os.path.join(dir_root,file) for file in files] #full path
    for file in files_paths:
        if os.path.isfile(file):#if file exists
            os.remove(file)
    if delete_art:
        try:
            file = os.path.join(dir_root,"art")
            shutil.rmtree(file)
        except:
            pass
   
def download_aid_update(tag):
    url_zip = "https://github.com/maikherbig/AIDeveloper/releases/download/"+tag+"/AIDeveloper_"+tag+".zip"
    path_save = "AIDeveloper_"+tag+".zip"
    path_save = os.path.join(dir_root,path_save)#path to save the update zip
    if not os.path.isfile(path_save):
        download_zip(url_zip,path_save)
        return {"success":True,"path_save":path_save}
    else:
        return {"success":False,"path_save":path_save}

def download_aid_repo():
    #Check online for most recent scripts (bleeding edge update)
    files = aideveloper_filelist()#+["art"]#TODO: also download the folder art
    url_scripts_repo = ["https://raw.github.com/maikherbig/AIDeveloper/master/AIDeveloper/"+f for f in files]
    path_temp = create_temp_folder()
    path_temp = os.path.join(path_temp,"update_files_bleeding_edge")
    
    try:
        if os.path.exists(path_temp):
            #delete this folder
            shutil.rmtree(path_temp)
        #Create this folder (nice and empty)
        os.mkdir(path_temp)
    
        for i  in range(len(files)):
            save_to = os.path.join(path_temp,files[i])
            download_zip(url_scripts_repo[i],save_to)
        
        #check completeness/integrity (certain scripts need to be there)
        check_aid_scripts_complete(path_temp)
        
        #create a zip file 
        date = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        path_save = os.path.join(dir_root,"AIDeveloper_"+date)
        shutil.make_archive(path_save, 'zip', path_temp)
        #delete the temporary files
        shutil.rmtree(path_temp)
        return {"success":True,"path_save":path_save+".zip"}
    except:
        return {"success":False,"path_save":""}

def update_from_zip(item_path,VERSION):
    #check that the zip contains all required files
    #zip_content = zipfile.ZipFile(item_path, 'r').namelist()
    
    #Unzip it into temp/update_files (only to check that everything is complete)
    #these files will actally be deleted immediately after checking
    path_temp = create_temp_folder()
    path_temp = os.path.join(path_temp,"update_files")
    extract_archive(item_path, path_temp, archive_format='zip')
    #check completeness/integrity (certain scripts need to be there)
    check_aid_scripts_complete(path_temp)
    #delete these files again
    shutil.rmtree(path_temp)

    #create a backup of the current version
    path_backup = backup_current_version(VERSION)
    #delete current version
    if item_path.endswith("update.zip"):#A release was downloaded. There, an art folder is contained
        #Delete all scripts and the art folder
        delete_current_version(delete_art=True)
    else:
        #Delete all scripts but keep the art folder
        delete_current_version(delete_art=False)
        
    #Unzip update into dir_root
    extract_archive(item_path, dir_root, archive_format='zip')

    #check completeness/integrity again (certain scripts need to be there)
    try:
        check_aid_scripts_complete(dir_root)
    except:#restore previous version from backup.zip 
        extract_archive(path_backup, dir_root, archive_format='zip')
    return path_backup




#import aid_start #import a module that sits in the AIDeveloper folder
#dir_root = os.path.dirname(aid_start.__file__)#ask the module for its origin

#def save_aid_settings(Default_dict):
#    dir_settings = os.path.join(dir_root,"aid_settings.json")#dir to settings
#    #Save the layout to Default_dict
#    with open(dir_settings, 'w') as f:
#        json.dump(Default_dict,f)



  

#################Some functions that are not used anymore######################

#def updates_online():
#    """
#    Searches the directory of AIDeveloper for available updates (zip folders)
#    Returns
#    list: list contains strings, each is a tag of an "a.b.c-update" version    
#    """
#    url_releases = "https://github.com/maikherbig/AIDeveloper/releases"
#    content = urllib.request.urlopen(url_releases).read().decode('UTF-8')
#    tags = content.split("/maikherbig/AIDeveloper/releases/tag/")[1:]
#    tags = [t.split('">AIDeveloper')[0] for t in tags]
#    tags_update = [a for a in tags if "update" in a]
#    tags_update = [a.split("-update")[0]+"-update" for a in tags_update]
#    return tags_update

#def zoom(a, factor):
#    a = np.asarray(a)
#    slices = [slice(0, old, 1/factor) for old in a.shape]
#    idxs = (np.mgrid[slices]).astype('i')
#    return a[tuple(idxs)]


#def load_model_json_h5(loc_json,loc_h5=False):
#    # load json and create model
#    json_file = open(loc_json, 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
#    if loc_h5:
#        # load weights into new model
#        loaded_model.load_weights(loc_h5)
#    return loaded_model

#class MyThread(QtCore.QThread):
#    def run(self):
#        self.exec_()


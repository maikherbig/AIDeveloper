# -*- coding: utf-8 -*-
"""
aid_bin
some useful functions that I want to keep separate to
make the main script a bit shorter
---------
@author: maikherbig
"""

import os,shutil,json,re,urllib
import numpy as np
import dclab
import h5py,time
import aid_start #import a module that sits in the AIDeveloper folder
dir_root = os.path.dirname(aid_start.__file__)#ask the module for its origin

def save_aid_settings(Default_dict):
    dir_settings = os.path.join(dir_root,"AIDeveloper_Settings.json")#dir to settings
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
            rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
        except:
            rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
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
                hash_new.append(rtdc_ds.hash)

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

def write_rtdc(fname,rtdc_datasets,X_valid,Indices,cropped=True,color_mode='Grayscale'):
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
            images_shape.append(rtdc_ds["image"].shape) 
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
                        images.append([rtdc_ds["image"][ii] for ii in indices])
                    pos_x.append([rtdc_ds["pos_x"][ii] for ii in indices])
                    pos_y.append([rtdc_ds["pos_y"][ii] for ii in indices])
                    
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
        hdf.create_dataset("events/pos_x", data=pos_x, dtype='uint8')
        hdf.create_dataset("events/pos_y", data=pos_y, dtype='uint8')
        hdf.create_dataset("events/index", data=index_new, dtype='uint8')
        
        #Adjust metadata:
        #"experiment:event count" = Nr. of images
        hdf.attrs["experiment:event count"]=images.shape[0]
        hdf.attrs["experiment:sample"]=fname
        hdf.attrs["imaging:pixel size"]=1.00
        hdf.attrs["experiment:event count"]=images.shape[0]
        hdf.close()
        return


    Features,Trace_lengths,Mask_dims_x,Mask_dims_y,Img_dims_x,Img_dims_y = [],[],[],[],[],[]
    for i in range(len(rtdc_datasets)):
        failed,rtdc_ds = load_rtdc(rtdc_datasets[i])
        if failed:
            print("Error occurred during loading file\n"+str(rtdc_datasets[i])+"\n"+str(rtdc_ds))
        else:
            features = rtdc_ds._events.keys()#all features
            Features.append(features)
    
            #The lengths of the fluorescence traces have to be equal, otherwise those traces also have to be dropped
            if "trace" in features:
                trace_lengths = [(rtdc_ds["trace"][tr][0]).size for tr in rtdc_ds["trace"].keys()]
                Trace_lengths.append(trace_lengths)
            #Mask Image dimensions have to be equal, otherwise those mask have to be dropped
            if "mask" in features:
                mask_dim = (rtdc_ds["mask"][0]).shape
                Mask_dims_x.append(mask_dim[0])
                Mask_dims_y.append(mask_dim[1])
            #Mask Image dimensions have to be equal, otherwise those mask have to be dropped
            if "image" in features:
                img_dim = (rtdc_ds["image"][0]).shape
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
                #get metadata of the dataset
                meta = {}
                # only export configuration meta data (no user-defined config)
                for sec in dclab.definitions.CFG_METADATA:
                    if sec in ["fmt_tdms"]:
                        # ignored sections
                        continue
                    if sec in rtdc_ds.config:
                        meta[sec] = rtdc_ds.config[sec].copy()
                        
                #Adjust the meta for the nr. of stored cells
                meta["experiment"]["event count"] = np.sum(np.array([len(indi) for indi in Indices])) 
                if cropped:
                    #Adjust the meta for cropped images
                    img_dim_x = Images[0].shape[1]
                    img_dim_y = Images[0].shape[0]
                    meta["imaging"]['roi size x'] = img_dim_x
                    meta["imaging"]['roi size y'] = img_dim_y
                
                #features = rtdc_ds._events.keys() #Get the names of the online features
                compression = 'gzip'    
                
                with dclab.rtdc_dataset.write_hdf5.write(path_or_h5file=fname,meta=meta, mode="append") as h5obj:
                    # write each feature individually
                    for feat in features:
                        # event-wise, because
                        # - tdms-based datasets don't allow indexing with numpy
                        # - there might be memory issues
                        if feat == "contour":
                            cont_list = [rtdc_ds["contour"][ii] for ii in indices]
                            dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                  data={"contour": cont_list},
                                  mode="append",
                                  compression=compression)
                        elif feat == "index":
                            dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                  data={"index": index_new_},
                                  mode="append",
                                  compression=compression)
                        elif feat in ["mask", "image"]:
                            # store image stacks (reduced file size and save time)
                            m = 64
                            if feat=='mask':
                                im0 = rtdc_ds[feat][0]
                            if feat=="image":
                                if cropped:
                                    im0 = Images[0]
                                else:
                                    im0 = rtdc_ds[feat][0]
                            imstack = np.zeros((m, im0.shape[0], im0.shape[1]),
                                               dtype=im0.dtype)
                            jj = 0
                            if feat=='mask':
                                image_list = [rtdc_ds[feat][ii] for ii in indices]
                            elif feat=='image':
                                if cropped:
                                    image_list = Images
                                else:
                                    image_list = [rtdc_ds[feat][ii] for ii in indices]
                            for ii in range(len(image_list)):
                                dat = image_list[ii]
                                if len(dat.shape)==3:#len(shape)=3 when there are multiple channels (RGB!)
                                    dat = (0.21 * dat[:,:,:1]) + (0.72 * dat[:,:,1:2]) + (0.07 * dat[:,:,-1:])
                                    dat = dat[:,:,0] 
                                    dat  = dat.astype(np.uint8)           
                                    if ii==0:
                                        print("Used Luminosity formula")
                                    
                                #dat = rtdc_ds[feat][ii]
                                imstack[jj] = dat
                                if (jj + 1) % m == 0:
                                    jj = 0
                                    dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                          data={feat: imstack},
                                          mode="append",
                                          compression=compression)
                                else:
                                    jj += 1
                            # write rest
                            if jj:
                                dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                      data={feat: imstack[:jj, :, :]},
                                      mode="append",
                                      compression=compression)
                        elif feat == "trace":
                            for tr in rtdc_ds["trace"].keys():
                                tr0 = rtdc_ds["trace"][tr][0]
                                trdat = np.zeros((len(indices), tr0.size), dtype=tr0.dtype)
                                jj = 0
                                trace_list = [rtdc_ds["trace"][tr][ii] for ii in indices]
                                for ii in range(len(trace_list)):
                                    trdat[jj] = trace_list[ii]
                                    jj += 1
                                dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                      data={"trace": {tr: trdat}},
                                      mode="append",
                                      compression=compression)
    
                        elif feat == "pos_x" and cropped==True:
                            data = np.zeros(shape=len(indices))+np.round(img_dim_x/2.0)*rtdc_ds.config["imaging"]["pixel size"]
                            dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                  data={feat: np.array(data)},mode="append")
                        elif feat == "pos_y" and cropped==True:
                            data = np.zeros(shape=len(indices))+np.round(img_dim_y/2.0)*rtdc_ds.config["imaging"]["pixel size"]
                            dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                  data={feat: np.array(data)},mode="append")
    
    
                        else:
                                data = [rtdc_ds[feat][ii] for ii in indices]
                                dclab.rtdc_dataset.write_hdf5.write(h5obj,
                                      data={feat: np.array(data)},mode="append")
                        
                    h5obj.close()

def check_update(this_version):
    """
    got to GitHub and check if there are any new releases available
    thisversion = version of the installed release
    """
    try:
        url_releases = "https://github.com/maikherbig/AIDeveloper/releases"
        content = urllib.request.urlopen(url_releases).read().decode('UTF-8')
        tags = content.split("/maikherbig/AIDeveloper/releases/tag/")[1:]
        tags = [t.split('">AIDeveloper')[0] for t in tags]
        tags_values =  [list(map(int, re.findall(r'\d+', t))) for t in tags]
        highest_major = np.max([t[0] for t in tags_values])
        highest_minor = np.max([t[1] for t in tags_values])
        highest_revis = np.max([t[2] for t in tags_values])
        tags_values = np.array(tags_values)
        #are there dev versions of the most recent version?
        latest_release = ".".join([str(highest_major),str(highest_minor),str(highest_revis)])
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
        
        dic = {"Errors":None,"latest_release":latest_release,"latest_release_url":url,"changelog":changelog}

        return dic
    
    except Exception as e:
        dic = {"Errors":e}
        #There is an some issue. Maybe not online...
        return dic


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

    

#################Some functions that are not used anymore######################

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


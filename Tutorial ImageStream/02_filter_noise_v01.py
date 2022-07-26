import os
import numpy as np
import h5py
import time

area_um_min, area_um_max = 100,500 # cell area in um
circ_min, circ_max = 0,1 #circularity (range: 0-1)
userdef1_min, userdef1_max = 0.5, 1 #probability for class 0 (range: 0-1)

# a. filter all files in a given folder
path = r"."
files = os.listdir(path)
filepaths = [os.path.join(path,file) for file in files]
# b. filter one specific file
filepaths = ["20220614_Cells_M04_299.rtdc"]

appendix = "_offset_"
invert_filter = True

def new_hdf(path_rtdc):
    hdf = h5py.File(path_rtdc,'w')
    #many attributes are required (so it can run in ShapeOut2 - but only for Grayscale images anyway...)
    hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
    hdf.attrs["experiment:event count"] = 100
    hdf.attrs["experiment:run index"] = 1
    hdf.attrs["experiment:sample"] = "Default"
    hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")
    hdf.attrs["imaging:flash device"] = "LED"
    hdf.attrs["imaging:flash duration"] = 2.0
    hdf.attrs["imaging:frame rate"] = 2000
    hdf.attrs["imaging:pixel size"] = 0.32
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
    hdf.attrs["setup:identifier"] = "Default"
    hdf.attrs["setup:medium"] = "CellCarrierB"
    hdf.attrs["setup:module composition"] = "AcCellerator"
    hdf.attrs["setup:software version"] = "dclab 0.9.0.post1 | dclab 0.9.1"
    return hdf


def store_trace(h5_group, data, indices):
    CHUNK_SIZE = 100
    keys_trace = list(data.keys())
    # create trace group
    #grp = h5obj.require_group("trace")

    for flt in keys_trace:
        # create traces datasets
        if flt not in h5_group:
            values = np.array(data[flt][:])
            values = values[indices]
            if len(values) == 1:# single event
                values = values.reshape(1, -1)
            maxshape = (None, values.shape[1])
            chunks = (CHUNK_SIZE, values.shape[1])
            h5_group.create_dataset(flt,data=values,maxshape=maxshape,
                               chunks=chunks,fletcher32=True,compression="gzip")
        else:
            dset = h5_group[flt]
            oldsize = dset.shape[0]
            values = np.array(data[flt][:])
            values = values[indices]
            dset.resize(oldsize + values.shape[0], axis=0)
            dset[oldsize:] = values



for file_path in filepaths:
    # load original file
    hdf_orig = h5py.File(filepaths[0],'r')
    # create new hdf file (path: path_target)
    path_target = file_path.split(".rtdc")[0]+appendix+".rtdc"    
    hdf_targ = new_hdf(path_target)
    
    #copy all the attributes from original ->target
    keys_attrs = hdf_orig.attrs.keys()
    for key_attrs in keys_attrs:
        hdf_targ.attrs[key_attrs] = hdf_orig.attrs[key_attrs]
        
    # load values from original file, for prospective filtering
    area = hdf_orig["events"]["area_um"][:]
    circ = 1.0 - hdf_orig["events"]["circ"][:]
    userdef0 = hdf_orig["events"]["userdef0"][:]
    userdef1 = hdf_orig["events"]["userdef1"][:]
    
    # get the filter-indexing
    ind = np.where((area>=area_um_min) & (area<=area_um_max) &
                   (circ>=circ_min) & (circ<=circ_max) &
                   (userdef1>=userdef1_min) & (userdef1<=userdef1_max) )[0]
    if invert_filter:
        ind_inv = np.zeros(area.shape, dtype='bool')+True
        ind_inv[ind] = False
        # replace the original filter index
        ind = ind_inv

    keys = list(hdf_orig["events"].keys())

    # create events group
    events = hdf_targ.require_group("events")

    for key in keys:
        #print("Writing: "+key)
        if key == "index":
            values = np.array(range(len(ind)))+1
            hdf_targ.create_dataset("events/"+key, data=values,dtype=values.dtype)

        elif key == "index_orig":
            values = hdf_orig["events"]["index"][ind]
            hdf_targ.create_dataset("events/"+key, data=values,dtype=values.dtype)

        elif "mask" in key:
            #print("omitting")
            mask = hdf_orig["events"][key][ind]
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.max() != 255 and mask.max() != 0 and mask.min() == 0:
                mask = mask / mask.max() * 255
            maxshape = (None, mask.shape[1], mask.shape[2])
            dset = hdf_targ.create_dataset("events/"+key, data=mask, dtype=np.uint8,maxshape=maxshape,fletcher32=True,chunks=True)
            dset.attrs.create('CLASS', np.string_('IMAGE'))
            dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
            dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))

        elif "image" in key:
            images = hdf_orig["events"][key][ind]
            maxshape = (None, images.shape[1], images.shape[2])
            dset = hdf_targ.create_dataset("events/"+key, data=images, dtype=np.uint8,maxshape=maxshape,fletcher32=True,chunks=True)
            dset.attrs.create('CLASS', np.string_('IMAGE'))
            dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
            dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                
        elif key == "trace":
            store_trace(h5group=events,
                        data=hdf_orig["events"]["trace"][ind],
                        compression="gzip")

        else:
            values = hdf_orig["events"][key][ind]
            hdf_targ.create_dataset("events/"+key, data=values,dtype=values.dtype)
    
    #Adjust metadata:
    #"experiment:event count" = Nr. of images
    hdf_targ.attrs["experiment:event count"] = len(ind)
    hdf_targ.close()
    hdf_orig.close()


    
    
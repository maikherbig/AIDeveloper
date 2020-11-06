
![alt text](Logo_AID_2_OpenCV.png "AID to OpenCV Logo")  

Along with this tutorial, all required materials/scripts will be provided to export models from AID and run them using OpenCV's dnn module. Since the OpenCV also offers the same API in C++, loading models and forwarding images would work equivalently.  
This tutorial is structured in a top-down fashion, meaning that you see first how a model is loaded and images are forwarded. The underlying code is explained afterwards. Only at the end, there are some tests and benchmarks.

# Run a model in OpenCV's dnn module   
```
import dclab,cv2,aid_cv2_dnn
import numpy as np

#Path to Smiley-Blink rtdc file (10 images of blink-smileys on noisy background)
rtdc_path = r"Smileys_Data\blink_10_gray.rtdc"
#Path to the frozen model
model_pb_path = r"Smileys_Models\MLP64_gray_9479_optimized.pb"
#Path to the meta file which was recorded when the model was trained
meta_path = r"Smileys_Models\MLP64_gray_meta.xlsx"

#Load the protobuf model
model_pb = cv2.dnn.readNet(model_pb_path)

#Extract image preprocessing settings from meta file
img_processing_settings = aid_cv2_dnn.load_model_meta(meta_path)

#Open the .rtdc file
rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
pix = rtdc_ds.config["imaging"]["pixel size"] #get pixelation (um/pix)
images = np.array(rtdc_ds["image"]) #get the images
pos_x,pos_y = rtdc_ds["pos_x"][:],rtdc_ds["pos_y"][:] 

#Compute the predictions
predictions = aid_cv2_dnn.forward_images_cv2(model_pb,img_processing_settings,
                                             images,pos_x,pos_y,pix)
```

# Installation  

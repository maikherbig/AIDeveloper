
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Logo_AID_2_OpenCV.png "AID to OpenCV Logo")  

Along with this tutorial, all required materials/scripts will be provided to export models from AID and run them using OpenCV's dnn module. Since the OpenCV also offers the same API in C++, loading models and forwarding images would work equivalently.  
This tutorial is structured in a top-down fashion, meaning you see first how a model is loaded and images are forwarded. The underlying code is explained afterwards. Only at the end, there are some tests and benchmarks.

# Run a model in OpenCV's dnn module   
```Python
import aid_cv2_dnn
import dclab,cv2
import numpy as np

# Path to Smiley-Blink rtdc file (10 images of blink-smileys on noisy background)
rtdc_path = r"Smileys_Data\blink_10_gray.rtdc"
# Path to the frozen model
model_pb_path = r"Smileys_Models\MLP64_gray_9479_optimized.pb"
# Path to the meta file which was recorded when the model was trained
meta_path = r"Smileys_Models\MLP64_gray_meta.xlsx"

# Load the protobuf model
model_pb = cv2.dnn.readNet(model_pb_path)

# Extract image preprocessing settings from meta file
img_processing_settings = aid_cv2_dnn.load_model_meta(meta_path)

# Open the .rtdc file
rtdc_ds = dclab.rtdc_dataset.RTDC_HDF5(rtdc_path)
pix = rtdc_ds.config["imaging"]["pixel size"] # pixelation (um/pix)
images = np.array(rtdc_ds["image"]) # get the images
pos_x, pos_y = rtdc_ds["pos_x"][:], rtdc_ds["pos_y"][:] 

# Compute the predictions
predictions = aid_cv2_dnn.forward_images_cv2(model_pb,img_processing_settings,
                                             images,pos_x,pos_y,pix)
```

The following paragraphs will cover:
- how images are preprocessed,
- how images are forwarded through a neural net
- how the smiley dataset was generated
- how the smiley classification model was trained
- how the model was exported
- tests for the image preprocessing functions of aid_cv2_dnn
  
# aid_cv2_dnn.py  
The script [aid_cv2_dnn.py](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/aid_cv2_dnn.py) 
contains all functions, required for preprocessing the .rtdc images 
and forwarding them through a neural net. For illustration, lets use the images
loaded above, which are are smileys, located at arbitrary positions on a noisy background:  
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Smiley_Blink_Examples_Gray.png "Smiley blink example images")  

It depends on the model, how image preprocessing is done. During training,
AIDevloper saves these settings in a meta file and those settings can be loaded using:
```Python
img_processing_settings = aid_cv2_dnn.load_model_meta(meta_path)
```
In the code block at the top of this page, predictions are obtained using
aid_cv2_dnn.forward_images_cv2. Within this function, image preprocessing is performed
using **aid_cv2_dnn.image_preprocessing**. This function carries out the following methods:  

- image_adjust_channels: adjust the number of channels of the images. Models in AIDeveloper can be trained using
grayscale or RGB images and the resulting model then expects images with either 1, or 3 channels.
If a grayscale image is provided, but the model expects 3 channels, the single channel of the image is 
copied three times. If an RGB image is provided but the model was trained using grayscale
images, the [luminosity method](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale)
is used to convert the RGB image into grayscale.  
- image_zooming: If images are captured using a magnification that is different from
the magnification that was used during capturing the training set, zooming allows to correct.
For example, lets say a model was trained using data that was captured using a 40x magnification.
If your imaging system offers just a 20x objective, the difference in magnification could 
be corrected by a zooming factor of 2.0. Keep in mind that objectives can result 
in images of unique phenotype. Especially  differences in numerical aperture (NA) can lead to 
differnt look of images and as a result the model will fail to predict correctly.
- image_crop_pad_cv2: crop images to particular size. In RT-DC, the location of 
objects is known and given by pos_x and pos_y. Images are cropped such that the
object is in the middle. If an object is too far at the border such that the 
desired image size cannot be obtained without going beyond the order of the image,
pixels are padded accordingly.
- image_normalization: 




The function forward_images_cv2, includes the 
image preprocessing step, which could also be done individually using

The shown images are RGB images (colorized). Both, RGB and grayscale images are
supported by the functions provided in aid_cv2_dnn. Given that a model was trained
using grayscale images, you can co



As in rtdc experiments, the position of the smiley in the image is known (pos_x, pos_y).
This allows to crop the image to the desired

Required image preprocessing steps are:
- adjusting the number of channels









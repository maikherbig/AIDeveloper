
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Logo_AID_2_OpenCV.png "AID to OpenCV Logo")  

Along with this tutorial, all required materials/scripts will be provided to 
export models from AID and run them using OpenCV's dnn module. Since the OpenCV 
also offers the same API in C++, loading models and forwarding images would work equivalently.  

[First, an example](#example) is provided which shows how to load an .rtdc file, load a frozen model,
perform image pre-processing and finally forward images throught the model.  
[Next, a step by step instruction](#step-by-step-instruction) is provided with more explanation of the underlying code.  
[Finally, tests and benchmarks](#tests-and-benchmarks) are provided. 

# Example

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

# Step by step instruction
The following paragraphs show how to deploy a model, step by step:
- [Export a model in AIDeveloper](#export-a-model-in-AIDeveloper) 
- [Pre-process images](#pre-process-images)
- [Forward images through neural net](#forward-images-through-neural-net)
  
## Export a model in AIDeveloper
1. Start AIDeveloper and go to the "History"-Tab.
2. Click the button 'Load model' on the lower left and choose a model that was trained earlier. 
3. Use the dropdown menu on the lower right and choose 'Optimized TensorFlow .pb'. 
4. Click the button 'Convert' on the lower right to run the conversion. After that you will find the correspoding model file in the same directory as the original model.  
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Export_Model_Combined_v01.png "Export Model")  
  
## Pre-process images 
The script [aid_cv2_dnn.py](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/aid_cv2_dnn.py) 
contains all functions, required for preprocessing images of an .rtdc file. 
It depends on the model, how image preprocessing is done and AIDeveloper saves these
settings during training in a so called meta file.  
You can load those settings using:
```Python
img_processing_settings = aid_cv2_dnn.load_model_meta(meta_path)
```
For image pre-processing, a dedicated function **aid_cv2_dnn.image_preprocessing**
carries out the followig methods:

- **image_adjust_channels**: adjust the number of channels of the images. Models in AIDeveloper can be trained using
grayscale or RGB images and the resulting model then expects images with either 1, or 3 channels.
If a grayscale image is provided, but the model expects 3 channels, the single channel of the image is 
copied three times. If an RGB image is provided but the model was trained using grayscale
images, the [luminosity method](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale)
is used to convert the RGB image into grayscale.  
- **image_zooming**: if images are captured using a magnification that is different from
the magnification that was used during capturing the training set, zooming allows to correct.
For example, lets say a model was trained using data that was captured using a 40x magnification.
If your imaging system offers just a 20x objective, the difference in magnification could 
be corrected by a zooming factor of 2.0. Keep in mind that objectives can result 
in images of unique phenotype. Especially  differences in numerical aperture (NA) can lead to 
differnt look of images and as a result the model will fail to predict correctly.
- **image_crop_pad_cv2**: crop images to particular size. In RT-DC, the location of 
objects is known and given by pos_x and pos_y. Images are cropped such that the
object is in the middle. If an object is too close to the border such that the 
desired image size cannot be obtained without going beyond the border of the image,
pixels are padded accordingly.
- **image_normalization**: This function carries out a normalization of the pixel
values.

Tests for that function are provedided [below](#test-image-preprocessing-test_image_preprocessing). 

## Forward images through neural net
To load an exported model, use:
```Python
model_pb = cv2.dnn.readNet(model_pb_path)
```
That model, along with the image pre-processing settings can now be used to 
forward images of an rtdc file:
```Python
predictions = aid_cv2_dnn.forward_images_cv2(model_pb,img_processing_settings,
                                             images,pos_x,pos_y,pix)
```
Tests for that function are provided [below](#test-model-inference-forward_images_cv2)). 

# Tests and benchmarks
Following paragraphs cover test functions which are contained in [aid_cv2_dnn_tests.py](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/aid_cv2_dnn_tests.py).  
Furthermore, details of the test-dataset are provided in [aid_cv2_dnn_tests.py](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/aid_cv2_dnn_tests.py).
- [Test image preprocessing (image_preprocessing)](#test-image-preprocessing-test_image_preprocessing)
- [Test model inference (forward_images_cv2)](#test-model-inference-forward_images_cv2)
- [Generation of the smiley dataset](#generation-of-the-smiley-dataset)
- Training the smiley classification model

## Test image preprocessing (test_image_preprocessing)
Irrespective of the location of the object, the function aid_cv2_dnn.image_preprocessing
needs to return an image of the desired target size witht the object in the middle.
To allow for a visual inspection, a smiley is placed on aribtrary positions on a noisy background.
The following conditions need to be tested.

Create grayscale (A) or RGB (B) images which reflect all possible phenotypes:     
- 1) raw image has odd width, but target image should have even width
- 2) raw image has odd height, but target image should have even height
- 3) raw image has odd width, and target image should also have odd width
- 4) raw image has odd height, and target image should also have odd height
  
for each of 1,2,3,4, test following conditions:
      
- a) cell far on the left
- b) cell far on the right
- c) cell far on top
- d) cell far on bottom

- f) target image wider than orignal image
- g) target image higher than orignal image
- h) target image wider and higher than orignal image

All tests can be carried out using the following command:
```Python
import aid_cv2_dnn_tests
aid_cv2_dnn_tests.test_image_preprocessing()
```
The successful test returns images of the desired size showing the sunglasses smiley in the center:  
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Test_ImagePreProcessing.png "Image Preprocessing Test")  


## Test model inference (forward_images_cv2)
The following steps allow to test the integrity of **forward_images_cv2** and the export function of 
AIDeveloper:
1. Use Keras to load the original model and compute predictions for some images
2. Use forward_images_cv2 to load the frozen model and compute predictions for the same images
3. compare both outputs

This logic is carried out by the function **aid_cv2_dnn_tests**, which you can find 
in [aid_cv2_dnn_tests.py](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/aid_cv2_dnn_tests.py).
To assure that forwarding images works for simple and also for advanced 
model architectures, two models were trained on the smiley dataset:
- multilayer perceptron with 3 layers  
- convolutional neural net with dropout and batch-normalization layers   
Furthermore, models may be trained using Grayscale or RGB images. For both options, models were
prepared using AIDeveloper. You can find the trained models in 
[Smileys_Models.zip](https://github.com/maikherbig/AIDeveloper/blob/master/Tutorial%20Deploy%20to%20OpenCV%20dnn/Smileys_Models.zip).

For each model architecture, forwarding images should work for
- Grayscale, and  
- RGB images.  

The respective test function is contained in aid_cv2_dnn_tests.py. Follwing script
uses that function to conduct all tests.

```Python
import aid_cv2_dnn_tests

#paths Smiley-Blink datasets (10 images) for grayscale and rgb images
datasets = [r"Smileys_Data\blink_10_gray.rtdc",r"Smileys_Data\blink_10.rtdc"]

for rtdc_path in datasets:
    # Simple MLP, trained on grayscale images
    meta_path = r"Smileys_Models\\MLP64_gray_meta.xlsx"#Path to the meta file which was recorded when the model was trained
    model_keras_path = r"Smileys_Models\\MLP64_gray_9479.model"#Path the the original model (keras hdf5 format)
    model_pb_path = r"Smileys_Models\\MLP64_gray_9479_optimized.pb"#Path to the frozen model
    # Run the test
    preds_mlp_gray = aid_cv2_dnn_tests.test_forward_images_cv2(rtdc_path,model_pb_path,meta_path,model_keras_path)
    
    # Simple MLP, trained on RGB images
    meta_path = r"Smileys_Models\\MLP64_rgb_meta.xlsx"#Path to the meta file which was recorded when the model was trained
    model_keras_path = r"Smileys_Models\\MLP64_rgb_9912.model"#Path the the original model (keras hdf5 format)
    model_pb_path = r"Smileys_Models\\MLP64_rgb_9912_optimized.pb"#Path to the frozen model
    # Run the test
    preds_mlp_rgb = aid_cv2_dnn_tests.test_forward_images_cv2(rtdc_path,model_pb_path,meta_path,model_keras_path)
    
    # CNN with dropout and batchnorm layers, trained on grayscale images
    meta_path = r"Smileys_Models\\LeNet_bn_do_gray_meta.xlsx"#Path to the meta file which was recorded when the model was trained
    model_keras_path = r"Smileys_Models\\LeNet_bn_do_gray_9259.model"#Path the the original model (keras hdf5 format)
    model_pb_path = r"Smileys_Models\\LeNet_bn_do_gray_9259_optimized.pb"#Path to the frozen model
    # Run the test
    preds_cnn_gray = aid_cv2_dnn_tests.test_forward_images_cv2(rtdc_path,model_pb_path,meta_path,model_keras_path)
    
    # CNN with dropout and batchnorm layers, trained on RGB images
    meta_path = r"Smileys_Models\\LeNet_bn_do_rgb_meta.xlsx"#Path to the meta file which was recorded when the model was trained
    model_keras_path = r"Smileys_Models\\LeNet_bn_do_rgb_9321.model"#Path the the original model (keras hdf5 format)
    model_pb_path = r"Smileys_Models\\LeNet_bn_do_rgb_9321_optimized.pb"#Path to the frozen model
    # Run the test
    preds_cnn_rgb = aid_cv2_dnn_tests.test_forward_images_cv2(rtdc_path,model_pb_path,meta_path,model_keras_path)
```



## Generation of the smiley dataset
Smileys of size 32x32 pixels showing 'blink', 'happy', or 'sunglasses' were placed at arbitrary positions on a noisy background:  
![alt text](https://github.com/maikherbig/AIDeveloper/blob/master/art/Smiley_Blink_Examples_Gray.png "Smiley blink example images")  







## Benchmarks
### Padding
AIDeveloper v<=0.1.2 uses np.pad, but OpenCV offers a similar implementation which 
should be preferred as it would also be available in C++. 
**aid_cv2_dnn_tests.pad_functions_compare** compares both functions and 

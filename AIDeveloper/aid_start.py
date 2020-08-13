# -*- coding: utf-8 -*-
"""
aid_bin.py
some functions that I want to keep separate to make the main script a bit shorter
---------
@author: maikherbig
"""

import os,shutil,json

def banner():
    #generated using: https://www.ascii-art-generator.org/    
    text = """
          _____ _____  
    /\   |_   _|  __ \ 
   /  \    | | | |  | |
  / /\ \   | | | |  | |
 / ____ \ _| |_| |__| |
/_/    \_\_____|_____/ 
is starting...
"""
    print(text)    
    


def get_default_dict(dir_settings):
    with open(dir_settings) as f:
        Default_dict = json.load(f)
        #Older versions of AIDeveloper might not have the Icon theme option->add it!
        if "Icon theme" not in Default_dict.keys():
            Default_dict["Icon theme"] = "Icon theme 1"
        if "Path of last model" not in Default_dict.keys():
            Default_dict["Path of last model"] = 'c:'+os.sep
        #Older versions of AIDeveloper might not have the Contrast, etc. Add it!
        if "Contrast On" not in Default_dict.keys():
            Default_dict["Contrast On"] = True
            Default_dict["Contrast min"] = 0.7
            Default_dict["Contrast max"] = 1.3
        if "Saturation On" not in Default_dict.keys():
            Default_dict["Saturation On"] = False
            Default_dict["Saturation min"] = 0.7
            Default_dict["Saturation max"] = 1.3
        if "Hue On" not in Default_dict.keys():
            Default_dict["Hue On"] = False
            Default_dict["Hue range"] = 0.08
        if "AvgBlur On" not in Default_dict.keys():
            Default_dict["AvgBlur On"] = True
            Default_dict["AvgBlur min"] = 0
            Default_dict["AvgBlur max"] = 5
        if "GaussBlur On" not in Default_dict.keys():
            Default_dict["GaussBlur On"] = False
            Default_dict["GaussBlur min"] = 0
            Default_dict["GaussBlur max"] = 5
        if "MotionBlur On" not in Default_dict.keys():
            Default_dict["MotionBlur On"] = False
            Default_dict["MotionBlur Kernel"] = "0,5"
            Default_dict["MotionBlur Angle"] = "-10,10"
        if "Image_import_dimension" not in Default_dict.keys():
            Default_dict["Image_import_dimension"] = 150
            Default_dict["Image_import_interpol_method"] = "Lanczos"

        if "spinBox_batchSize" not in Default_dict.keys():
            Default_dict["spinBox_batchSize"] = 32
        if "doubleSpinBox_learningRate_Adam" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_Adam"] = 0.001
        if "doubleSpinBox_learningRate_SGD" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_SGD"] = 0.01
        if "doubleSpinBox_learningRate_RMSprop" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_RMSprop"] = 0.001
        if "doubleSpinBox_learningRate_Adagrad" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_Adagrad"] = 0.01
        if "doubleSpinBox_learningRate_Adadelta" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_Adadelta"] = 1.0
        if "doubleSpinBox_learningRate_Adamax" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_Adamax"] = 0.002
        if "doubleSpinBox_learningRate_Nadam" not in Default_dict.keys():
            Default_dict["doubleSpinBox_learningRate_Nadam"] = 0.002




    return Default_dict

def save_default_dict(dir_settings):
    #In case the standard dictionary is overwritten, this function can create a
    #initial disctionary that works
    #Users can open the .json with a text editor and change stuff if needed
    Default_dict = {"norm_methods":["None","Div. by 255","StdScaling using mean and std of each image individually","StdScaling using mean and std of all training data"],\
    "Input image size":32,"Normalization":"Div. by 255",\
    "Nr. epochs":2500,"Keras refresh after nr. epochs": 2,\
    
    "Image_import_dimension":150,"Image_import_interpol_method":"Lanczos",\
    
    "Horz. flip":False,"Vert. flip":True,"rotation":3,\
    "width_shift":0.001,"height_shift":0.001,"zoom":0.001,"shear":0.005,\
    
    "Brightness refresh after nr. epochs": 1,\
    "Brightness add. lower":-15,"Brightness add. upper":15,\
    "Brightness mult. lower":0.7,"Brightness mult. upper":1.3,\
    
    "Gaussnoise Mean":0,"Gaussnoise Scale":3.0,\
    
    "Contrast On": True, "Contrast min":0.7, "Contrast max":1.3,\
    "Saturation On": False, "Saturation min":0.7, "Saturation max":1.3,\
    "Hue On": False, "Hue range":0.08,\
    
    "AvgBlur On": True, "AvgBlur min":0, "AvgBlur max":5,\
    "GaussBlur On": False, "GaussBlur min":0, "GaussBlur max":5,\
    "MotionBlur On": False, "MotionBlur Kernel":"0,5", "MotionBlur Angle":"-10,10",\
    
    "Layout": "Normal", "Icon theme":"Icon theme 1",\
    "Path of last model":"c:\\",\

    "spinBox_batchSize":128,\
    "doubleSpinBox_learningRate_Adam":0.001,\
    "doubleSpinBox_learningRate_SGD":0.01,\
    "doubleSpinBox_learningRate_RMSprop":0.001,\
    "doubleSpinBox_learningRate_Adagrad":0.01,\
    "doubleSpinBox_learningRate_Adadelta":1.0,\
    "doubleSpinBox_learningRate_Adamax":0.002,\
    "doubleSpinBox_learningRate_Nadam":0.002}

    with open(dir_settings, 'w') as f:
        json.dump(Default_dict,f)




def keras_json_replace(keras_json_path,json_exists=True):
    if json_exists:
        #Inform user!
        print("I found a keras.json file in your home-directory which has options AID does not accept.\
              This file will be copied to keras_beforeAID_x.json and a new keras.json\
              is written with valid options for AID.")
        i=0
        while os.path.isfile(os.path.expanduser('~')+os.sep+'.keras'+os.sep+'keras_beforeAID_'+str(i)+'.json'):
            i+=1
        shutil.copy(keras_json_path, os.path.expanduser('~')+os.sep+'.keras'+os.sep+'keras_beforeAID_'+str(i)+'.json')
    
    #Write new keras.json:
    with open(os.path.expanduser('~')+os.sep+'.keras'+os.sep+'keras.json','w') as f:
        new_settings = """{\n    "image_dim_ordering": "tf", \n    "epsilon": 1e-07, \n    "floatx": "float32", \n    "backend": "tensorflow"\n}"""                       
        f.write(new_settings)

def keras_json_check(keras_json_path):
    if os.path.isfile(keras_json_path):
        with open(keras_json_path, 'r') as keras_json:
            keras_json=keras_json.read()
        keras_json = json.loads(keras_json)
        keys_keras_json = keras_json.keys() #contained keys 
        keys_expected = ['image_dim_ordering','backend','epsilon','floatx'] #expected keys
        #are those keys present in keys_keras_json?
        keys_present = [key in keys_expected for key in keys_keras_json]
        keys_present = all(keys_present) #are all true?
        checks = []
        if keys_present==True:#all keys are there, now check if the value is correct
            checks.append( keras_json['image_dim_ordering']=="tf" )   
            checks.append( keras_json['backend']=="tensorflow" )    
            checks.append( keras_json['epsilon']==1e-07 )   
            checks.append( keras_json['floatx']=="float32" )    
            checks = all(checks) #are all true?
            if checks==True:
                #keras.json is fine, no need to overwrite it
                return
            else:#some values are different. AID need to write new keras.json
                keras_json_replace(keras_json_path)
        
        else:#some keys are missing
            keras_json_replace(keras_json_path)
    
    else:#there exists NO keras.json! Very likely the user opened AID for the first time :)
        #Welcome the user
        print("A warm welcome to your first session in AID :)\
             \nIn case of any issues just write me a mail: maik.herbig@tu-dresden.de\
             \nor use the Issues section on the github page of AID:\
             \nhttps://GitHub.com/maikherbig/AIDeveloper\
             ")
        keras_json_replace(keras_json_path,False)

def get_tooltips():
    tooltips = {}
    tooltips["groupBox_dragdrop"] = "<html><head/><body><p>Drag and drop files (.rtdc) or folders with images here. Valid .rtdc files have to contain at least 'images', 'pos_x' and 'pos_y'. If folders with images are dropped, the contents will be converted to a single .rtdc file (speeds up loading in the future).<br>After dropping data, you can specify the ‘Class’ of the images and if it should be used for training (T) or validation (V).<br>Double-click on the filename to show a random image of the data set. The original and cropped (using 'Input image size') image is shown<br>Click on the button 'Plot' to open a popup where you can get a histogram or scatterplot of enclosed data.<br>'Cells/Epoch' defines the nr. of images that are used in each epoch for training. Random images are drawn in each epoch during training. For the validation set, images are drawn once at the beginning and kept constant.<br>Deactivate 'Shuffle' if you don't want to use random data. Then all images of this file are used.<br>Zoom allows you to increase or decrease resolution. Zoom=1 does nothing; Zoom=2 zooms in; Zoom=0.5 zooms out. This is useful if you have data acquired at 40x but you want to use it to train a model for 20x data. Use 'Options'->'Zooming order' to define the method used for zooming.<br>Hint for RT-DC users: Use ShapeOut to gate for particular subpopulations and save the filtered data as .rtdc. Make sure to export at least 'images', 'pos_x' and 'pos_y'.</p></body></html>"
    tooltips["groupBox_DataOverview"] = "<html><head/><body><p>The numbers of events of each class are added up. To do so the properties of each file are read. This happens each time you click into the table above. Unfortunately, reading is quite slow, so maybe disable this box, while you are assembling your data-set (especially when working with many large files). Use the column 'Name' to specify custom class-names, which help you later to remember, the meaning of each class. This table is saved to meta.xlsx when training is started.</p></body></html>"
    tooltips["tab_ExampleImgs"] = "<html><head/><body><p>Show random example images of the training data</p></body></html>"
    tooltips["comboBox_ModelSelection"] = "<html><head/><body><p>Select the model architecture. MLP means Multilayer perceptron. Currently only MLPs are fast enough for AI based sorting. The definition of the architectures can be found in the model_zoo.py. If you want to implement custom Neural net architectures, you can edit model_zoo.py accordingly. Restart AIDeveloper in order to re-load model_zoo and import the new definitions</p></body></html>"
    tooltips["radioButton_NewModel"] = "<html><head/><body><p>Select a model architecture in the dropdown menu</p></body></html>"
    tooltips["radioButton_LoadRestartModel"] = "<html><head/><body><p>Load an existing architecture (from .model or .arch), and start training using random initial weights</p></body></html>"
    tooltips["radioButton_LoadContinueModel"] = "<html><head/><body><p>Load an existing model (.model) and continue fitting. This option could be used to load a trained model to optimize it (transfer learning)</p></body></html>"
    tooltips["lineEdit_LoadModelPath"] = "<html><head/><body><p>When you use 'Load and restart' or 'Load and continue', the filename of the chosen model will apprear here</p></body></html>"
    tooltips["label_Crop"] = "<html><head/><body><p>Models need a defined input size image. Choose wisely since cutting off from large objects might result in information loss.</p></body></html>"
    tooltips["label_Normalization"] = "<html><head/><body><p>Image normalization method.<br><br>None: No normalization is applied.<br><br>'Div. by 255': Each input image is divided by 255 (useful since pixelvalues go from 0 to 255, so the result will be in range 0-1).<br><br>StdScaling using mean and std of each image individually: The mean and standard deviation of each input image itself is used to scale it by first subtracting the mean and then dividing by the standard deviation.<br><br>StdScaling using mean and std of all training data: First, all pixels of the entire training dataset are used to calc. a mean and a std. deviation. This mean and standard deviation is used to scale images during training by first subtracting the mean and then dividing by the standard deviation.<br><br>Only 'None' and 'Div. by 255' are currently supported in the Sorting-Software</p></body></html>"
    tooltips["label_colorMode"] = "<html><head/><body><p>The Color Mode defines the input image depth. Color images have three channels -RGB. Grayscale images only have a single channel. Models are automatically built accordingly. Models trained for Grayscale cannot be applied to RGB images (unless images are converted). Same is true the other way around.</p></body></html>"
    tooltips["label_nrEpochs"] = "<html><head/><body><p>Number of epochs (iterations over the training data). In each epoch, a subset of the training data is used to update the weights if 'Shuffle' is on.</p></body></html>"
    tooltips["label_paddingMode"] = "<html><head/><body><p>Padding mode to use when requested image size is larger than available images.<br>constant (default): Pads with a constant value.<br>edge: Pads with the edge values of array.<br>linear_ramp: Pads with the linear ramp between end_value and the array edge value.<br>maximum: Pads with the maximum value of all or part of the vector along each axis.<br>mean: Pads with the mean value of all or part of the vector along each axis.<br>median: Pads with the median value of all or part of the vector along each axis.<br>minimum: Pads with the minimum value of all or part of the vector along each axis.<br>reflect: Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.<br>symmetric: Pads with the reflection of the vector mirrored along the edge of the array.<br>wrap: Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.<br>Text copied from https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html</p></body></html>"
    tooltips["pushButton_modelname"] = "<html><head/><body><p>Define path and filename for the model you want to fit.</p></body></html>"

    tooltips["radioButton_cpu"] = "<html><head/><body><p>Train model on CPU.</p></body></html>"
    tooltips["radioButton_gpu"] = "<html><head/><body><p>Train model on GPU.</p></body></html>"
    tooltips["comboBox_gpu"] = "<html><head/><body><p>Select the device your model should be trained on. 'Multi-GPU' will train the model on multiple GPUs in parallel.</p></body></html>"

    
    tooltips["label_memory"] = "<html><head/><body><p>Limit the amount of memory used. 1.0 means 100%; 0.5 means 50%; Typically, 0.7 is a good value for a system with one GPU.</p></body></html>"
    tooltips["label_RefreshAfterEpochs"] = "<html><head/><body><p>Affine image augmentation takes quite long; so maybe use the same images for this nr. of epochs</p></body></html>"
    tooltips["tab_kerasAug"] = "<html><head/><body><p>Define settings for  affine image augmentations</p></body></html>"
    tooltips["checkBox_HorizFlip"] = "<html><head/><body><p>Flip some training images randomly along horiz. axis (left becomes right; right becomes left)</p></body></html>"
    tooltips["checkBox_VertFlip"] = "<html><head/><body><p>Flip some training images randomly along vert. axis (bottom up; top down)</p></body></html>"
    tooltips["label_Rotation"] = "<html><head/><body><p>Degree range for random rotations</p></body></html>"
    tooltips["label_width_shift"] = "<html><head/><body><p>Define random shift of width<br>Fraction of total width, if &lt; 1. Otherwise pixels if>=1.<br>Value defines an interval (-width_shift_range, +width_shift_range) from which random numbers are created.</p></body></html>"
    tooltips["label_height_shift"] = "<html><head/><body><p>Define random shift of height<br>Fraction of total height if &lt; 1. Otherwise pixels if>=1.<br>Value defines an interval (-height_shift_range, +height_shift_range) from which random numbers are created.</p></body></html>"
    tooltips["label_zoom"] = "<html><head/><body><p>Range for random zoom</p></body></html>"
    tooltips["label_shear"] = "<html><head/><body><p>Shear Intensity (Shear angle in counter-clockwise direction in degrees)</p></body></html>"
    tooltips["spinBox_RefreshAfterEpochs"] = "<html><head/><body><p>Affine image augmentation takes quite long; so maybe use the same images for this nr. of epochs</p></body></html>"
    tooltips["label_RefreshAfterNrEpochs"] = "<html><head/><body><p>Brightness augmentation is fast, so you can probably refresh images for each epoch (set to 1)</p></body></html>"
    tooltips["groupBox_BrightnessAugmentation"] = "<html><head/><body><p>Define add/multiply offset to make image randomly slightly brighter or darker. Additive offset (A) is one number that is added to all pixels values; Multipl. offset (M) is a value to multiply each pixel value with: NewImage = A + M*Image</p></body></html>"
    tooltips["label_Plus"] = "<html><head/><body><p>Brightness augmentation by adding a random value from given range.</p></body></html>"
    tooltips["spinBox_PlusLower"] = "<html><head/><body><p>Define lower threshold for additive offset</p></body></html>"
    tooltips["spinBox_PlusUpper"] = "<html><head/><body><p>Define upper threshold for additive offset</p></body></html>"
    tooltips["label_Mult"] = "<html><head/><body><p>Brightness augmentation by multiplying the pixel values of the image with a given value (random value from given range).</p></body></html>"

    tooltips["doubleSpinBox_MultLower"] = "<html><head/><body><p>Define lower threshold for multiplicative offset</p></body></html>"
    tooltips["doubleSpinBox_MultUpper"] = "<html><head/><body><p>Define upper threshold for multiplicative offset</p></body></html>"
    tooltips["groupBox_GaussianNoise"] = "<html><head/><body><p>Define Gaussian Noise, which is added to the image</p></body></html>"
    tooltips["label_GaussianNoiseMean"] = "<html><head/><body><p>Define the mean of the Gaussian noise. Typically this should be zero. If you use a positive number it would mean that your noise tends to be positive, i.e. bright.</p></body></html>"
    tooltips["label_GaussianNoiseScale"] = "<html><head/><body><p>Define the standard deviation of the Gaussian noise. A larger number means a wider distribution of the noise, which results in an image that looks more noisy. Prefer to change this parameter over chainging the mean.</p></body></html>"
    tooltips["groupBox_colorAugmentation"] = "<html><head/><body><p>Define methods to randomly alter the contrast (applicable for grayscale and RGB), saturation (RGB only) or hue (RGB only) of your images.</p></body></html>"
    tooltips["checkBox_contrast"] = "<html><head/><body><p>Augment contrast using a random factor. Applicable for Grayscale and RGB. Left spinbox (lower factor) has to be >0. '0.70' to '1.3' means plus/minus 30% contrast (at random)</p></body></html>"
    tooltips["checkBox_saturation"] = "<html><head/><body><p>Augment saturation using a random factor. Applicable for RGB only. Left spinbox (lower factor) has to be >0. '0.70' to '1.3' means plus/minus 30% saturation (at random)</p></body></html>"
    tooltips["checkBox_hue"] = "<html><head/><body><p>Augment hue using a random factor. Applicable for RGB only. Left spinbox (lower factor) has to be >0. '0.70' to '1.3' means plus/minus 30% hue (at random)</p></body></html>"
    tooltips["groupBox_blurringAug"] = "<html><head/><body><p>Define methods to randomly blur images.</p></body></html>"
    tooltips["label_motionBlurKernel"] = "<html><head/><body><p>Define kernels by giving a range [min,max]. Values in this range are then randomly picked for each image</p></body></html>"
    tooltips["lineEdit_motionBlurAngle"] = "<html><head/><body><p>Define angle for the motion blur by defining a range \"min degree,max degree\". Values in this range are then randomly picked for each image</p></body></html>"
    tooltips["label_avgBlurMin"] = "<html><head/><body><p>Define the minimum kernel size for average blur</p></body></html>"
    tooltips["spinBox_gaussBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for gaussian blur</p></body></html>"
    tooltips["checkBox_motionBlur"] = "<html><head/><body><p>Apply random motion blurring. Motion blur is defined by an intensity ('kernel') and a direction ('angle'). Please define a range for 'kernel' and 'angle'. AID will pick a random value (within each range) for each image.</p></body></html>"
    tooltips["spinBox_avgBlurMin"] =  "<html><head/><body><p>Define the minimum kernel size for average blur</p></body></html>"
    tooltips["spinBox_gaussBlurMin"] = "<html><head/><body><p>Define the minimum kernel size for gaussian blur</p></body></html>"
    tooltips["label_motionBlurAngle"] = "<html><head/><body><p>Define a range of angles for the motion blur: 'min degree,max degree'. Values from this range are then randomly picked for each image</p></body></html>"
    tooltips["label_gaussBlurMin"] = "<html><head/><body><p>Define the minimum kernel size for gaussian blur</p></body></html>"
    tooltips["checkBox_gaussBlur"] = "<html><head/><body><p>Apply random gaussian blurring. For gaussian blurring, a gaussian kernel of defined size is used. Please define a min. and max. kernel size. For each image a random value is picked from this range to generate a gaussian kernel.</p></body></html>"
    tooltips["spinBox_avgBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for average blur</p></body></html>"
    tooltips["label_gaussBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for gaussian blur</p></body></html>"
    tooltips["label_gaussBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for gaussian blur</p></body></html>"
    tooltips["checkBox_avgBlur"] = "<html><head/><body><p>Apply random average blurring. Define a range of kernel sizes for the average blur (min. and max. kernel size). Values from this range are then randomly picked for each image. To blur an image, all pixels within the kernel area used to compute an average pixel value. The central element of the kernel area in the image is then set to this value. This operation is carried out over the entire image.</p></body></html>"
    tooltips["label_avgBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for average blur</p></body></html>"
    tooltips["spinBox_avgBlurMin"] = "<html><head/><body><p>Define the minimum kernel size for average blur</p></body></html>"
    tooltips["spinBox_avgBlurMax"] = "<html><head/><body><p>Define the maximum kernel size for average blur</p></body></html>"
    tooltips["lineEdit_motionBlurKernel"] = "<html><head/><body><p>Define kernels by giving a range \"min,max\". Values from this range are then randomly picked for each image</p></body></html>"
    tooltips["groupBox_expertMode"] = "<html><head/><body><p>Expert mode allows changing the learning rate and you can even set parts of the neural net to \'not trainable\' in order to perform transfer learning and fine tune models. Also dropout rates can be adjusted. When expert mode is turned off again, the initial values are applied again.</p></body></html>"
    tooltips["groupBox_learningRate"] = "<html><head/><body><p>Change the learning rate. The default optimizer is \'adam\' with a learning rate of 0.001</p></body></html>"
    tooltips["checkBox_trainLastNOnly"] = "<html><head/><body><p>When checked, only the last n layers of the model, which have >0 parameters will stay trainable. Layers before are set to trainable=False. Please specify n using the spinbox. After this change, the model has to be recompiled, which means the optimizer values are deleted.</p></body></html>"  
    tooltips["spinBox_trainLastNOnly"] = "<html><head/><body><p>Specify the number of last layer in your model that should be kept trainable. Only layers with >0 parameters are counted. Use the checkbox to apply this option. After this change, the model has to be recompiled, which means the optimizer values are deleted.</p></body></html>"
    tooltips["checkBox_trainDenseOnly"] = "<html><head/><body><p>When checked, only the dense layers are kept trainable (if they have >0 parameters). Other layers are set to trainable=False. After this change, the model has to be recompiled, which means the optimizer values are deleted.</p></body></html>"
    tooltips["label_batchSize"] = "<html><head/><body><p>Number of samples per gradient update. If unspecified, batch_size will default to 32. (Source: Keras documentation)</p></body></html>"
    tooltips["label_epochs"] = "<html><head/><body><p>Number of epochs to train the model on an identical batch.</p></body></html>"    
 
    tooltips["groupBox_learningRate"] = "<html><head/><body><p>The learning rate defines how strong parameters are changed in each training iteration.</p></body></html>"    
    tooltips["radioButton_LrConst"] = "<html><head/><body><p>Define a constant learning rate.</p></body></html>"    
    tooltips["doubleSpinBox_learningRate"] = "<html><head/><body><p>Define a constant value for the learning rate. The learning rate defines how strong parameters are changed in each training iteration.</p></body></html>"    
    tooltips["radioButton_LrCycl"] = "<html><head/><body><p>Apply cyclical learning rate schedule. Here, the learning rate oszillates between two bounds (Min/Max). After each processed batch, the lr is adjusted. Button (...) leads to functions that allow to find sensible bounds. Cyclical learning rate was orignally defined in the paper 'Cyclical Learning Rates for Training Neural Networks': https://arxiv.org/abs/1506.01186.</p></body></html>"    
    tooltips["label_cycLrMin"] = "<html><head/><body><p>Lower bound for cyclical learning rate.</p></body></html>"    
    tooltips["label_cycLrMax"] = "<html><head/><body><p>Upper bound for cyclical learning rate.</p></body></html>"    
    tooltips["comboBox_cycLrMethod"] = "<html><head/><body><p>Method for changing the learning rate.</p></body></html>"    
    tooltips["pushButton_cycLrPopup"] = "<html><head/><body><p>Methods to find sensible bounds for cyclicyal learning rates.</p></body></html>"    
    
    tooltips["radioButton_LrExpo"] = "<html><head/><body><p>Apply exponentially decreasing learning rates. Equation: LR = initial_LR * decay_rate ^ (epoch / decay_steps).</p></body></html>"    
    tooltips["label_expDecInitLr"] = "<html><head/><body><p>Apply exponentially decreasing learning rates. Equation: LR = initial_LR * decay_rate ^ (epoch / decay_steps).</p></body></html>"    
    tooltips["label_expDecSteps"] = "<html><head/><body><p>Decay steps: Nr. of epochs it should take till LR decreases to decay_rate*inital_LR. Equation: LR = initial_LR * decay_rate ^ (epoch / decay_steps).</p></body></html>"    
    tooltips["label_expDecRate"] = "<html><head/><body><p>Decay rate: choose value between 0 and 1. Lower means earlier drop of LR. 1 means never decrease learning rate.</p></body></html>"    

    tooltips["checkBox_dropout"] = "<html><head/><body><p>If your model has one or more dropout layers, you can change the dropout rates here. Insert into the lineEdit one value (e.g. 0.5) to apply this one value to all dropout layers, or insert a list of values to specify the dropout rates for each dropout layer individually (e.g. for three dropout layers: [ 0.2 , 0.5 , 0.25 ]. The model will be recompiled, but the optimizer weights are not deleted.</p></body></html>"
    tooltips["checkBox_partialTrainability"] = "<html><head/><body><p>Partial trainability allows you to make parts of a layer non-trainable. Hence, this option makes most sense in combination with 'Load and continue' training a model. After checking this box, the model you chose on 'Define model'-tab is initialized. The line on the right shows the trainability of each layer in the model. Use the button on the right ('...') to open a popup menu, which allows you to specify individual trainabilities for each layer.</p></body></html>"
    tooltips["label_expt_loss"] = "<html><head/><body><p>Specify which loss function should be used. From Keras documentation: 'The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.'</p></body></html>"
    tooltips["label_optimizer"] = "<html><head/><body><p>Specify which optimizer should be used.'</p></body></html>"

    tooltips["checkBox_lossW"] = "<html><head/><body><p>Specify scalar coefficients to weight the loss contributions of different classes.</p></body></html>"
    tooltips["groupBox_expertMetrics"] = "<html><head/><body><p>Define metrics, that are computed after each training iteration ('epoch'). Those metrics are can then also be displayed in real-time during training and are tracked/saved in the meta.xlsx file. Each model where any metric for the validation set breaks a new record is saved (minimum val. loss achived -> model is saved. maximum val. accuracy achieved-> model is saved).</p></body></html>"
    tooltips["checkBox_expertAccuracy"] = "<html><head/><body><p>Compute accuracy and validation accuracy after each epoch. Each model, where the corresponding metric for the validatio-set achieves a new record will be saved.</p></body></html>"
    tooltips["checkBox_expertF1"] = "<html><head/><body><p>Compute F1 and validation F1 score after each epoch. Each model, where the corresponding metric for the validatio-set achieves a new record will be saved.</p></body></html>"
    tooltips["checkBox_expertPrecision"] = "<html><head/><body><p>Compute precision and validation precision after each epoch for each class. Each model, where the corresponding metric for the validatio-set achieves a new record will be saved.</p></body></html>"
    tooltips["checkBox_expertRecall"] = "<html><head/><body><p>Compute recall and validation recall after each epoch for each class. Each model, where the corresponding metric for the validatio-set achieves a new record will be saved.</p></body></html>"
    tooltips["pushButton_FitModel"] = "<html><head/><body><p>Afer defining all model parameters, hit this button to build/initialize the model, load the data to RAM (if 'Load data to RAM' is chosen in 'Edit') and start the fitting process. You can also do only the initialization to check the model summary (appears in textbox on the left).</p></body></html>"
    tooltips["pushButton_Live"] = "<html><head/><body><p>Load and display the history of the model which is currently fitted</p></body></html>"
    tooltips["pushButton_LoadHistory"] = "<html><head/><body><p>Select a history file to be plotted</p></body></html>"
    tooltips["lineEdit_LoadHistory"] = "<html><head/><body><p>Enter path/filename of a meta-file (meta.xlsx). The history is contained in this file.</p></body></html>"
    tooltips["tableWidget_HistoryItems"] = "<html><head/><body><p>Information of the history file is shown here<br>Double-click to adjust color</p></body></html>"
    tooltips["checkBox_rollingMedian"] = "<html><head/><body><p>Check to add a rolling median. Use the slider to change the window size for which the median is computed.</p></body></html>"
    tooltips["horizontalSlider_rollmedi"] = "<html><head/><body><p>Use this slider to change the window size for the rolling median between 1 and 50.</p></body></html>"
    tooltips["checkBox_linearFit"] = "<html><head/><body><p>Check if you want to add a liner fit. A movable region will appear. Only data inside this region will be used for the fit.</p></body></html>"
    tooltips["pushButton_LoadModel"] = "<html><head/><body><p>Load a model from disk for conversion to other formats. Please specify the format of the model using the dropbox above. Next, define the target format using the dropdown menu on the right-> and finally, hit 'Convert'.</p></body></html>"
    tooltips["pushButton_convertModel"] = "<html><head/><body><p>Convert chosen model to the format indicated by the dropbox above. This might be useful to deply models to other platforms. AIDeveloper is only compatible with Keras TensorFlow models to perform inference. For usage of the model with soRT-DC, please convert it to .nnet (currently only MLPs are supported by soRT-DC software!).</p></body></html>"
    tooltips["lineEdit_ModelSelection_2"] = "<html><head/><body><p>Model architecture name, read from meta.xlsx ('Chosen Model') is displayed here</p></body></html>"
    tooltips["tableWidget_Info_2"] = "<html><head/><body><p>Specify validation data via 'Build'-tab or load .rtdc file (->From .rtdc).<br>Use column 'Name' to specify proper cell names (for presentation purposes).<br>Use column 'clr' to specify plotting color of that cell-type</p></body></html>"
    tooltips["lineEdit_LoadModel_2"] = "<html><head/><body><p>Enter path/filename of a model (.model)</p></body></html>"
    tooltips["pushButton_ExportValidToNpy"] = "<html><head/><body><p>Export the validation data (images and labels). Optionally the cropped images can exported->use 'Options'->'Export' to change. Normalization method of the chosen model is not yet applied. Please use the \'Build\' tab to define data</p></body></html>"
    tooltips["pushButton_ImportValidFromNpy"] = "<html><head/><body><p>Import validation data (images from .rtdc and labels from .txt) file. Cropped and non-cropped images can be imported. If necessary they will be cropped to the correct format. If the loaded images are smaller than the required size, there will be zero-padding.</p></body></html>"
    tooltips["groupBox_InferenceTime"] = "<html><head/><body><p>Inference time is the time required to predict a single image. To get a meaningful value, several (define how many using spinbox->) images are predicted one by one. The given Nr. (spinbox->) is divided by 10. The resulting nr. of images is predicted one by one and an average computing time is obtained. This process is repqeated 10 times</p></body></html>"
    tooltips["label_SortingIndex"] = "<html><head/><body><p>Specify the class you intend to sort for ('Target cell type'). This is important if you want to know the final concentration in the target region</p></body></html>"
    tooltips["checkBox_SortingThresh"] = "<html><head/><body><p>Specify a probability threshold above which a cell is classified as a target cell (specified using 'Sorting class'). Typically cells with a probability above 0.5 are sorted, but you could also increase the threshold in order to only have cells in the target region that are more confidently classified</p></body></html>"
    tooltips["comboBox_probability_histogram"] = "<html><head/><body><p>Select a plotting style for the probability plot<br>Style1: Only outline, width 5<br>Style2: Only outline, width 10<br>Style4: Filled hist, alpha 0.6<br>Style3: Filled hist, alpha 0.7<br>Style5: Filled hist, alpha 0.8</p></body></html>"
    tooltips["groupBox_3rdPlot"] = "<html><head/><body><p>ROC (Receiver Operating Characteristic) curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.<br>Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.<br>ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.</p></body></html>"
    tooltips["comboBox_3rdPlot"] = "<html><head/><body><p>Use the combobox to define what is shown in the third plot. Some options might only be valid for binary problems. For such cases please use the spinboxes Indx1 and Indx2 to define two cell types</p></body></html>"
    tooltips["label_Indx1"] = "<html><head/><body><p>Some options of the combobox (\'Third plot\') are only valid for binary problems. For such cases please use the spinboxes Indx1 and Indx2 to define two cell types</p></body></html>"
    tooltips["tableWidget_CM1"] = "<html><head/><body><p>Confusion matrix shows the total Nrs. of cells. Doubleclick a field in the matrix to save the corresponding images to .rtdc</p></body></html>"
    tooltips["tableWidget_CM2"] = "<html><head/><body><p>Normalized Confusion matrix shows the relative amount of cells. Doubleclick a field in the matrix to save the corresponding images to .rtdc </p></body></html>"
    text_scikit_learn = "Compute precision, recall, F-measure and support for each class<br>\
    The precision is the ratio tp / (tp + fp) where tp is the number of \
    true positives and fp the number of false positives. The precision is \
    intuitively the ability of the classifier not to label as positive a \
    sample that is negative.<br>The recall is the ratio tp / (tp + fn) where \
    tp is the number of true positives and fn the number of false negatives. \
    The recall is intuitively the ability of the classifier to find all the \
    positive samples.<br>The F-beta score can be interpreted as a weighted \
    harmonic mean of the precision and recall, where an F-beta score reaches \
    its best value at 1 and worst score at 0. The F-beta score weights recall \
    more than precision by a factor of beta. beta == 1.0 means recall and \
    precision are equally important.<br>The support is the number of occurrences \
    of each class in y_true.<br>micro average - averaging the total true \
    positives false negatives and false positives<br>macro average - averaging \
    the unweighted mean per label<br>weighted average - averaging the \
    support-weighted mean per label<br>(Source:scikit-learn.org)"
    tooltips["tableWidget_AccPrecSpec"] = "<html><head/><body><p>Classification Metrics appear after hitting 'Update Plots'<br>Copy to clipboad by clicking somewhere on table<br><br>"+text_scikit_learn+"</p></body></html>"
    tooltips["groupBox_probHistPlot"] = "<html><head/><body><p>Probability histogram appears after hitting 'Prob. hist'. It shows the predicted probability of each cell to remain to class 'Sorting clas'. Colors indicate the true label. A good model returns high probabilities for cells of 'Sorting class', while cells of other indices (different cell types) have very low probabilities. This plot also allows to guess a reasonable 'Sorting threshold'</p></body></html>"
    tooltips["comboBox_chooseRtdcFile"] = "<html><head/><body><p>Choose a file</p></body></html>"
    tooltips["comboBox_featurey"] = "<html><head/><body><p>Dropdown menu shows all availble features of the chosen .rtdc-file. The chosen feature will be plotted on the y-axis</p></body></html>"
    tooltips["widget_histx"] = "<html><head/><body><p>Histogram projection of the x-dimension</p></body></html>"
    tooltips["widget_histy"] = "<html><head/><body><p>Histogram projection of the y-dimension</p></body></html>"   
    tooltips["horizontalSlider_cellInd"] = "<html><head/><body><p>Use this slider to choose a cell in the data-set. The respective cell and trace will be shown in tab 'Peakdetection'</p></body></html>"  
    tooltips["spinBox_cellInd"] = "<html><head/><body><p>Use this to index and choose an event in the data-set. The respective cell and trace will be shown in tab 'Peakdetection'</p></body></html>"
    tooltips["widget_scatter"] = "<html><head/><body><p>Click into this scatterplot to choose an event. The respective cell and trace will be shown in tab 'Peakdetection'</p></body></html>"
    tooltips["checkBox_fl1"] = "<html><head/><body><p>Check this if you want to plot the trace for Fl1. Automatic peak finding (Highest x% in 'Peakdetection'-tab) will then also search in those traces.</p></body></html>"
    tooltips["checkBox_fl2"] = "<html><head/><body><p>Check this if you want to plot the trace for Fl2. Automatic peak finding (Highest x% in 'Peakdetection'-tab) will then also search in those traces.</p></body></html>"
    tooltips["checkBox_fl3"] = "<html><head/><body><p>Check this if you want to plot the trace for Fl3. Automatic peak finding (Highest x% in 'Peakdetection'-tab) will then also search in those traces.</p></body></html>"
    tooltips["checkBox_centroid"] = "<html><head/><body><p>Check this if you want to plot the centroid of a chosen event in 'Show cell' on 'Peakdetection'-tab).</p></body></html>"
    tooltips["pushButton_selectPeakPos"] = "<html><head/><body><p>After you moved the range on the trace-plot you can select a peak using this button. This data of the peak will be shown in the table in the right and will be used to fit the peak-detection model.</p></body></html>"
    tooltips["pushButton_selectPeakRange"] = "<html><head/><body><p>After you changed the range on the trace-plot you can select the range-width using this button. This range will be shown in the table below and will be used in the peak-detection model.</p></body></html>"
    tooltips["pushButton_highestXPercent"] = "<html><head/><body><p>The highest x% of FLx_max peaks are looked up for each x (=1,2,3 if Checkboxes are activated) and inserted into the table on the right.</p></body></html>"
    tooltips["pushButton_removeSelectedPeaks"] = "<html><head/><body><p>Select a peak in the table or in the plot on the right and remove it using this button.</p></body></html>"    
    tooltips["pushButton_removeAllPeaks"] = "<html><head/><body><p>Remove all peaks from the table the right.</p></body></html>"
    tooltips["widget_showSelectedPeaks"] = "<html><head/><body><p>Scatterplot shows all selcted peaks. Click on a peak to highlight the respective position in the table. After that you can also use the button 'Selected' to remove this point</p></body></html>"
    tooltips["tableWidget_showSelectedPeaks"] = "<html><head/><body><p>Table shows all seelcted peaks. After clicking on a row you can use the button 'Selected' to remove this point</p></body></html>"
    tooltips["groupBox_showCell"] = "<html><head/><body><p>Plot shows image of the recorded cell and the respective fluorescence traces (optional- use checkboxes to show FL1/2/3). Optionally the centroid is shown.</p></body></html>"
    tooltips["pushButton_updateScatterPlot"] = "<html><head/><body><p>Hit this button to read the chosen features and plot the scatterplot above.</p></body></html>"
    tooltips["tableWidget_peakModelParameters"] = "<html><head/><body><p>After fitting a peak-detection model, the model parameters are shown here. Each parameter can also be manipulated right here.</p></body></html>"
    tooltips["comboBox_peakDetModel"] = "<html><head/><body><p>Choose a peak detection model.</p></body></html>"
    tooltips["pushButton_fitPeakDetModel"] = "<html><head/><body><p>Start fitting a model usig the selected peaks shown above</p></body></html>"
    tooltips["pushButton_SavePeakDetModel"] = "<html><head/><body><p>Save model to an excel file</p></body></html>"
    tooltips["pushButton_loadPeakDetModel"] = "<html><head/><body><p>Load peak detection model from an excel file</p></body></html>"
    tooltips["radioButton_exportSelected"] = "<html><head/><body><p>Apply the peak detection model only on the single chosen file</p></body></html>"
    tooltips["radioButton_exportAll"] = "<html><head/><body><p>Apply the peak detection model on all files on the 'Build'-Tab</p></body></html>"
    tooltips["modelsaved_success"] = "<html><head/><body><p>The model was successfully saved. 'Load and continue' was automatically selected  in 'Define Model'-Tab, so your model with partial trainability will be loaded when you start fitting. The model architecture is documented in each .model file and the .arch file. Change of partial trainability during training is not supported yet (but it is theoretically no problem to implement it).</p></body></html>"


    ###############Fitting popup window##################
    tooltips["checkBox_realTimePlotting_pop"] = "<html><head/><body><p>Plot model metrics like accuracy, val. accuracy... in real time. Please first hit 'Update plot' to initiate the plotting region.</p></body></html>"
    tooltips["label_realTimeEpochs_pop"] = "<html><head/><body><p>Define how many of the last epochs should be plotted in real time. 250 means the last 250 epochs are plotted</p></body></html>"
    tooltips["pushButton_clearTextWindow_pop"] = "<html><head/><body><p>Clear the text window (fitting info).</p></body></html>"
    tooltips["checkBox_ApplyNextEpoch"] = "<html><head/><body><p>Changes made in this window will be applied at the next epoch.</p></body></html>"
    tooltips["checkBox_saveEpoch_pop"] = "<html><head/><body><p>Save the model, when the current epoch is done</p></body></html>"
    tooltips["pushButton_Pause_pop"] = "<html><head/><body><p>Pause fitting, push this button again to continue.</p></body></html>"
    tooltips["pushButton_Stop_pop"] = "<html><head/><body><p>Stop fitting entirely, Close this window manually, after the progressbar shows 100%.</p></body></html>"
    tooltips["actioncpu_merge"] = "<html><head/><body><p>Identify whether to force merging model weights under the scope of the CPU or not. Source: https://www.tensorflow.org/api_docs/python/tf/keras/utils/multi_gpu_model</p></body></html>"
    tooltips["actioncpu_relocation"] = "<html><head/><body><p>Identify whether to create the model's weights under the scope of the CPU. If the model is not defined under any preceding device scope, you can still rescue it by activating this option. Source: https://www.tensorflow.org/api_docs/python/tf/keras/utils/multi_gpu_model</p></body></html>"
    tooltips["actioncpu_weightmerge"] = "<html><head/><body><p>Uses tf.device('/cpu:0') prior constriction model to manage merging of weights using CPU.</p></body></html>"
    
    ############Confusion matrix show image popup
    tooltips["groupBox_model"] = "<html><head/><body><p>Properties of the model.</p></body></html>"
    tooltips["pushButton_showSummary"] = "<html><head/><body><p>Show a textual summary of your model.</p></body></html>"
    tooltips["label_inpImgSize"] = "<html><head/><body><p>Input image size.</p></body></html>"
    tooltips["label_outpSize"] = "<html><head/><body><p>Output dimension of model. Typically, the number of classes.</p></body></html>"
    tooltips["pushButton_toTensorB"] = "<html><head/><body><p>Show model architecture in Tensorboard (web browser will be started).</p></body></html>"
    tooltips["groupBox_imageShow"] = "<html><head/><body><p>Our image will be shown here.</p></body></html>"
    tooltips["groupBox_image_Settings"] = "<html><head/><body><p>Plotting options.</p></body></html>"
    tooltips["label_image_alpha"] = "<html><head/><body><p>Alpha value for the image.</p></body></html>"
    tooltips["groupBox_gradCAM_Settings"] = "<html><head/><body><p>Settings for Grad-CAM activation heatmap. Title of original paper: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization; URL: https://arxiv.org/abs/1610.02391 </p></body></html>"
    tooltips["label_gradCAM_targetClass"] = "<html><head/><body><p>Determine the class for which the activation heatmap should be shown.</p></body></html>"
    tooltips["label_gradCAM_targetLayer"] = "<html><head/><body><p>Determine the layer for which the activation heatmap should be show. Typically, the last convolutional layer is used.</p></body></html>"
    tooltips["label_gradCAM_colorMap"] = "<html><head/><body><p>Colormap for the Grad-CAM activation heatmap.</p></body></html>"
    tooltips["label_gradCAM_alpha"] = "<html><head/><body><p>Alpha value for the Grad-CAM heatmap.</p></body></html>"
    tooltips["pushButton_reset"] = "<html><head/><body><p>Reset settings.</p></body></html>"
    tooltips["pushButton_update"] = "<html><head/><body><p>Apply the settings and update the image.</p></body></html>"
    tooltips["groupBox_model"] = "<html><head/><body><p>Information about the model for which learning rates should be screened.</p></body></html>"
    tooltips["label_startLR"] = "<html><head/><body><p>Define the lower bound for screening learning rates.</p></body></html>"
    tooltips["label_stopLr"] = "<html><head/><body><p>Define the upper bound for screening learning rates.</p></body></html>"
    tooltips["label_percData"] = "<html><head/><body><p>Carrying out the screening for learning rates, requires to actually fit the model for some epochs. In case your loaded dataset is huge that might take very long. With this option here, you can choose to only use a subset of your data to speed up the process.</p></body></html>"
    tooltips["label_stepsPerEpoch"] = "<html><head/><body><p>The number of screening steps per epoch is calculated using nr_training_images / batch_size.</p></body></html>"
    tooltips["pushButton_LrReset"] = "<html><head/><body><p>Reset the LR screening settings to initial values.</p></body></html>"
    tooltips["pushButton_color"] = "<html><head/><body><p>Click to open a menu to choose a color for the LR screening plot.</p></body></html>"
    tooltips["label_lineWidth"] = "<html><head/><body><p>Define the width of the line in the LR screening plot.</p></body></html>"
    tooltips["label_epochs"] = "<html><head/><body><p>Number of epochs to train. Higher will give more precise result. Typically, 3-5 epochs are sufficient.</p></body></html>"
    tooltips["pushButton_LrFindRun"] = "<html><head/><body><p>Run a LR screening. This might take some minutes since data has to be loaded and a model is actually trained for some epochs. Note that all parameters (image augmentation etc) are used as indicated in the corresponding menus. Hence, make sure to first set these values as it affects training.</p></body></html>"
    tooltips["groupBox_LrSettings"] = "<html><head/><body><p>Carry out a screening of different learning rates as introduced in https://arxiv.org/abs/1506.01186.<br>Very low learning rates will not allow the model to learn anything and the resulting loss stays constant. Too high learning rates will cause large updates of model weights which prevents finding a minumum. You may consult this page to learn more how to choose good learning rates and how it is implemented:<br>https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder </p></body></html>"    
    tooltips["groupBox_singleLr"] = "<html><head/><body><p>When activating the box, a line will appear in the plot above, which you can drag to define a single learning rate value. You should choose a point where the loss is very low (minimum), but does yet start to fluctuate.</p></body></html>"
    tooltips["groupBox_LrRange"] = "<html><head/><body><p>When activating this box, a range will appear in the plot above, which you can drag to define a range of learning rates. The left edge should be located where the loss starts to decrease and the right edge close to the minumum.</p></body></html>"
    tooltips["pushButton_LR_plot"] = "<html><head/><body><p>Plot the learning rate vs epochs. The plot depends on the parameters defined in this menu. The plot allows you to decide whether you like the leanring rate schedule.</p></body></html>"



    ####deprecated!
    tooltips["checkBox_learningRate"] = "<html><head/><body><p>Change the learning rate. The default optimizer is \'adam\' with a learning rate of 0.001</p></body></html>"

    return tooltips

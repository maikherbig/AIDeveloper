# -*- coding: utf-8 -*-
"""
aideveloper_bin
some useful functions that I want to keep separate to
make the main script a bit shorter
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
    "Path of last model":"c:\\"}

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


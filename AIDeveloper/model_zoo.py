# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:36:16 2018

@author: maikh
"""
from keras.models import Sequential,Model
from keras.layers import Add,Concatenate,Input,Dense,Dropout,Flatten,Activation,Conv2D,MaxPooling2D,BatchNormalization,GlobalAveragePooling2D
from keras.layers.merge import add
from keras import regularizers
import keras_applications
import keras
import numpy as np


version = "0.0.9_Original" #1.) Use any string you like to specify/define your version of the model_zoo
def __version__():
    #print(version)
    return version

#2.) Insert the name of your model in this list. All those names here, will appear later in the GUI
predefined_1 = ["MLP_4_4_4", "MLP_8_8_8", "MLP_16_8_16", "MLP_24_16_24", "MLP_64_32_16", "MLP_24_16_24_skipcon","MLP_256_128_64_do"]
predefined_2 = ["LeNet5","LeNet5_do","LeNet5_bn_do","LeNet5_bn_do_skipcon","TinyResNet","TinyCNN"]
predefined_3 = ["VGG_small_1", "VGG_small_2","VGG_small_3","VGG_small_4"]
predefined_4 = ["Nitta_et_al_6layer","Nitta_et_al_8layer","Nitta_et_al_6layer_linact"]
predefined_5 = ["MhNet1_bn_do_skipcon","MhNet2_bn_do_skipcon","CNN_4Conv2Dense_Optim"]
predefined_6_1 = ["pretrained_mobilenet_v2","pretrained_nasnetmobile","pretrained_nasnetlarge","pretrained_densenet","pretrained_mobilenet","pretrained_inception_v3","pretrained_vgg19","pretrained_vgg16","pretrained_xception"]
predefined_6_2 = ["pretrained_resnet50","pretrained_resnet101","pretrained_resnet152","pretrained_resnet50_v2","pretrained_resnet101_v2","pretrained_resnet152_v2"]
predefined_7 = ["Collection_1","Collection_2","Collection_3","Collection_4","Collection_5","Collection_6"]
multiInputModels = ["MLP_MultiIn_3Lay_64"]
predefined_coll_test = ["Collection_test"]

predefined_models = predefined_1+predefined_2+predefined_3+predefined_4+\
predefined_5+predefined_6_1+predefined_6_2+predefined_7+predefined_coll_test+multiInputModels
def get_predefined_models(): #this function is used inside AIDeveloper.py
    return predefined_models

#3.) Update the get_model function
def get_model(modelname,in_dim,channels,out_dim):
    if modelname.startswith("MLP") and "skipcon" not in modelname and "do" not in modelname and "MultiIn" not in modelname :
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim,channels,out_dim)
    elif modelname=="MLP_24_16_24_skipcon":
        model = MLP_24_16_24_skipcon(in_dim,channels,out_dim)
    elif modelname=="MLP_256_128_64_do":
        model = MLP_256_128_64_do(in_dim,channels,out_dim)
    elif modelname=="MLP_MultiIn_3Lay_64":
        model = MLP_MultiIn_3Lay_64(in_dim,channels,out_dim)

    elif modelname=="LeNet5":
        model = LeNet5(in_dim,channels,out_dim)
    elif modelname=="LeNet5_do":
        model = LeNet5_do(in_dim,channels,out_dim)
    elif modelname=="LeNet5_bn_do":
        model = LeNet5_bn_do(in_dim,channels,out_dim)
    elif modelname=="LeNet5_bn_do_skipcon":
        model = LeNet5_bn_do_skipcon(in_dim,channels,out_dim)
    elif modelname=="TinyResNet":
        model = TinyResNet(in_dim,channels,out_dim)

    elif modelname=="TinyCNN":
        model = TinyCNN(in_dim,channels,out_dim)
    elif modelname=="VGG_small_1":
        model = VGG_small_1(in_dim,channels,out_dim)
    elif modelname=="VGG_small_2":
        model = VGG_small_2(in_dim,channels,out_dim)
    elif modelname=="VGG_small_3":
        model = VGG_small_3(in_dim,channels,out_dim)
    elif modelname=="VGG_small_4":
        model = VGG_small_4(in_dim,channels,out_dim)

    elif modelname=="Nitta_et_al_6layer":
        model = nitta_et_al_6layer(in_dim,in_dim,channels,out_dim)

    elif modelname=="Nitta_et_al_8layer":
        model =  nitta_et_al_8layer(in_dim,in_dim,channels,out_dim)
    elif modelname=="Nitta_et_al_6layer_linact":
        model =  Nitta_et_al_6layer_linact(in_dim,in_dim,channels,out_dim)
        
    elif modelname=="MhNet1_bn_do_skipcon":
        model =  MhNet1_bn_do_skipcon(in_dim,channels,out_dim)
    elif modelname=="MhNet2_bn_do_skipcon":
        model =  MhNet2_bn_do_skipcon(in_dim,channels,out_dim)
    elif modelname=="CNN_4Conv2Dense_Optim":
        model =  CNN_4Conv2Dense_Optim(in_dim,channels,out_dim)
        
    elif modelname=="pretrained_mobilenet_v2":
        model =  pretrained_mobilenet_v2(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_nasnetmobile":
        model =  pretrained_nasnetmobile(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_nasnetlarge":
        model =  pretrained_nasnetlarge(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_densenet":
        model =  pretrained_densenet(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_mobilenet":
        model =  pretrained_mobilenet(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_inception_v3":
        model =  pretrained_inception_v3(in_dim,in_dim,channels,out_dim)

    elif modelname=="pretrained_vgg19":
        model =  pretrained_vgg19(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_vgg16":
        model =  pretrained_vgg16(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_xception":
        model =  pretrained_xception(in_dim,in_dim,channels,out_dim)

    elif modelname=="pretrained_resnet50":
        model =  pretrained_resnet50(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_resnet101":
        model =  pretrained_resnet101(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_resnet152":
        model =  pretrained_resnet152(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_resnet50_v2":
        model =  pretrained_resnet50_v2(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_resnet101_v2":
        model =  pretrained_resnet101_v2(in_dim,in_dim,channels,out_dim)
    elif modelname=="pretrained_resnet152_v2":
        model =  pretrained_resnet152_v2(in_dim,in_dim,channels,out_dim)

    elif modelname=="Collection_1":
        model =  collection_1(in_dim,in_dim,channels,out_dim)
    elif modelname=="Collection_2":
        model =  collection_2(in_dim,in_dim,channels,out_dim)
    elif modelname=="Collection_3":
        model =  collection_3(in_dim,in_dim,channels,out_dim)
    elif modelname=="Collection_4":
        model =  collection_4(in_dim,in_dim,channels,out_dim)
    elif modelname=="Collection_5":
        model =  collection_5(in_dim,in_dim,channels,out_dim)
    elif modelname=="Collection_6":
        model =  collection_6(in_dim,in_dim,channels,out_dim)

    return model

#4. Define your model architecture:
#Rules: 
#- specify the name of the first layer as "inputTensor" (see examples below)
#- specify the name of the last layer as "outputTensor" (see examples below)
#- AIDeveloper currently only supports single-input (image) - single-output (prediction) models.
#- have a look at the example neural nets below if you like (they will not 
#appear in AIDeveloper since they are not described in "predefined_models" and "get_model")
    
#########################Example Neural nets###################################
#Example 1: Sequential API
def Example_sequential_api(in_dim,channels,out_dim):
    model = Sequential()
    
    model.add(Conv2D(16,kernel_size=3,strides=3,padding='same',input_shape=(in_dim, in_dim,channels),name="inputTensor"))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(8,kernel_size=3,strides=3,padding='same'))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  #this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    
    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])
    return model

#Example 2: Functional API
def Example_functional_api(in_dim,channels,out_dim):
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format (height,width,channels)
    
    x = Conv2D(16,kernel_size=3,strides=3,padding='same')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(8,kernel_size=3,strides=3,padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)  #this converts the 3D feature maps to 1D feature vectors

    x = Dense(32)(x)
    x = Activation('relu')(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model


################################MLPs###########################################

def mlp_generator(model_string,in_dim,channels,out_dim):
    model_string = model_string.split("_")
    nodes_in_layer = []
    for layer in range(len(model_string)):
        try:
            nodes = int(model_string[layer])
            nodes_in_layer.append(nodes)
        except:
            pass        
    nr_dense_layers = len(nodes_in_layer)
    model = Sequential()
    model.add(Flatten(input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow

    for i in range(nr_dense_layers):
        model.add(Dense(nodes_in_layer[i])) 
        model.add(Activation('relu'))
    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

    return model
 
    
def mlpconfig_to_str(model_config):
    #if it is an MLP, pop first and last layer
    text = "MLP"
    ismlp = 1
    l = model_config["config"]
    if l[0]["class_name"]=='Flatten':
        for i in list(np.array(range(len(l)))[1::2])[:-1]:
            if l[i]["class_name"]=="Dense":
                if l[i+1]["class_name"]=="Activation":
                    #Get the number of nodes in that layer
                    nodes = l[i]["config"]["units"]
                    text+="_"+str(nodes)
                else:
                    ismlp=0
            else:
                ismlp=0
        if l[-2]["class_name"]=="Dense":
            if l[-1]["class_name"]=="Activation":
                out_dim = l[-2]["config"]["units"]
            else:
                ismlp=0
        else:
            ismlp=0
        return ismlp,text
   
    
def MLP_24_16_24_skipcon(in_dim,channels,out_dim):
    #Define an MLP with skip connections
    #This requries functional API!
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution

    x = Flatten()(inputs)  # this converts the 3D feature maps to 1D feature vectors
    x = Dense(24)(x)
    x_keep = Activation('relu')(x)
    
    x = Dense(16)(x_keep)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Add()([x,x_keep])#Skip Connection
        
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model

def MLP_256_128_64_do(in_dim,channels,out_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow

    model.add(Dense(256)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(64)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])
    return model


################################CNNs###########################################

def LeNet5(in_dim,channels,out_dim):
    model = Sequential()
    
    #model.add(Conv2D(6,5,5,input_shape=(1,in_dim, in_dim))) #Theano
    model.add(Conv2D(6,5,5,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, 5,5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(120))
    model.add(Activation('relu'))
    
    model.add(Dense(84))
    model.add(Activation('relu'))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model


def LeNet5_do(in_dim,channels,out_dim):
    model = Sequential()
    
    #model.add(Conv2D(6,5,5,input_shape=(1,in_dim, in_dim))) #Theano
    model.add(Conv2D(6,5,5,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, 5,5))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model



def LeNet5_bn_do(in_dim,channels,out_dim):
    model = Sequential()
    
    #model.add(Conv2D(6,5,5,input_shape=(1,in_dim, in_dim))) #Theano
    model.add(Conv2D(6,5,5,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(BatchNormalization()) #add axis=1 for thenao and -1(default) for Tensorflow
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, 5,5))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model


def LeNet5_bn_do_skipcon(in_dim,channels,out_dim):
    #Define a CNN similar to the LeNet but with skip connections (ResNet)
    #This requries functional API!
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution

    inputs_mp = MaxPooling2D(pool_size=(2, 2))(inputs) #create a maxpooled version to allow skipconnection also after maxpooling
    inputs_mp = Conv2D(16,kernel_size=1, padding='same')(inputs_mp)
    
    x = Conv2D(6,kernel_size=(5,5), padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    #inbetween1 = Conv2D(kernel_size=1, filters=6, strides=1, padding='same')(x)

    #x = Add()([x,inbetween1])#Skip Connection
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    inbetween2 = Conv2D(kernel_size=1, filters=16, strides=1, padding='same')(x)
    
    x = Conv2D(16,kernel_size=(5,5), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Add()([x,inputs_mp])#Skip Connection
    x = Add()([x,inbetween2])# #Also add inbetween (Skip Connection)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors
    
    x = Dense(120)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(84)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model

def MhNet1_bn_do_skipcon(in_dim,channels,out_dim):
    #Define a CNN similar to the LeNet but with skip connections (ResNet)
    #This requries functional API!
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution

    x = Conv2D(8,kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    inbetween1 = MaxPooling2D(pool_size=(2, 2))(x) #for the add layer

    x = Conv2D(16,kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    inbetween2 = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32,kernel_size=(3,3), padding='same')(inbetween2)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(16,kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Add()([x,inbetween2])#Skip Connection

    x = Conv2D(8,kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x,inbetween1])#Skip Connection
    x = Dropout(0.05)(x)

    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors
    
    x = Dense(120)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(84)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model



def MhNet2_bn_do_skipcon(in_dim,channels,out_dim):
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution
    inputs_bn = BatchNormalization()(inputs)
    inputs_bn = Flatten()(inputs_bn)  # this converts the 3D feature maps to 1D feature vectors    
    inputs_bn = Dense(256)(inputs_bn)
    inputs_bn = Activation('relu')(inputs_bn)
       
    x = Conv2D(32,kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    inbetween1_conv = Conv2D(kernel_size=1, filters=32, strides=1, padding='same')(x)
    inbetween1_f = Flatten()(x)
    inbetween1_f_d = Dense(16)(inbetween1_f)

    x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    inbetween2_conv = Conv2D(kernel_size=1, filters=32, strides=1, padding='same')(x)
    inbetween2_f = Flatten()(x)
    inbetween2_f_d = Dense(16)(inbetween2_f)

    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!
    x = Dropout(0.25)(x)

    x = Conv2D(32,kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Add()([x,inbetween1_conv])#Skip Connection

    x = Conv2D(32,kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Add()([x,inbetween2_conv])#Skip Connection
    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!
    x = Dropout(0.25)(x)

    inbetweens = Concatenate()([inbetween1_f_d,inbetween2_f_d])#Skip Connection
    inbetweens = Activation('relu')(inbetweens)
    inbetweens = BatchNormalization()(inbetweens)
    inbetweens = Dropout(0.5)(inbetweens)

    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors
    
    x = Dense(256-32)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Concatenate()([x,inbetweens])#Skip Connection
    x = Add()([x,inputs_bn])#Skip Connection

    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model


def CNN_4Conv2Dense_Optim(in_dim,channels,out_dim):
    """
    I run a screening using 32x32 images of retina cells and altered c1,
    c2 and p. In total 772 architectures were screened and this one here
    outperformed all others
    """
    c1 = 6
    c2 = 36
    p = 0.4

    model = Sequential()
    model.add(Conv2D(c1,3,3,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))

    model.add(Conv2D(c1,3,3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    
    model.add(Conv2D(c2,5,5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))

    model.add(Conv2D(c2,7,7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(p))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(p))
    
    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

    return model


#Helper function for TinyResNet
def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        # H_l(x):
        h = BatchNormalization()(x) # first pre-activation
        h = Activation("relu")(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation("relu")(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x) # 1x1 conv2d
        else:
            f = x # identity
        
        
        return add([f, h]) # F_l(x) = f(x) + H_l(x):
    return f

def TinyResNet(in_dim,channels,out_dim):
    #Source: https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
    #Author: Egor Malykh
    input_tensor = Input((in_dim, in_dim,channels),name="inputTensor")
    
    # first conv2d with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=24, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # F_1
    x = block(24)(x)
    # F_2
    x = block(24)(x)
    
    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)
    
    # dropout for more robust learning
    x = Dropout(0.2)(x)
    
    # last softmax layer
    x = Dense(units=out_dim, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation("softmax")(x)
    
    model = Model(inputs=input_tensor, outputs=x)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def TinyCNN(in_dim,channels,out_dim):
    model = Sequential()
    
    #model.add(Conv2D(6,5,5,input_shape=(1,in_dim, in_dim))) #Theano
    model.add(Conv2D(3,3,3,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(3,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(16))
    model.add(Activation('relu'))
    
    model.add(Dense(24))
    model.add(Activation('relu'))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model



def VGG_small_1(in_dim,channels,out_dim):
    model = Sequential()
    
    model.add(Conv2D(32,3,3,input_shape=(in_dim, in_dim,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])
    return model

def VGG_small_2(in_dim,channels,out_dim):
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution
    inputs_bn = BatchNormalization()(inputs)
    inputs_bn = Conv2D(kernel_size=1, filters=64, strides=1, padding='valid')(inputs_bn)
    inputs_bn = MaxPooling2D(pool_size=(3, 3))(inputs_bn) 
    
    x = Conv2D(32,kernel_size=3,strides=1, padding='valid')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!
    x = BatchNormalization()(x)

    x = Conv2D(64,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)

    x = Add()([x,inputs_bn])#Skip Connection

    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors
    x = BatchNormalization()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model

def VGG_small_3(in_dim,channels,out_dim): #Very similar to VGG_small_2, but just less BN layers and one more dense layer
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution
    inputs_ = Conv2D(kernel_size=1, filters=64, strides=1, padding='valid')(inputs)
    inputs_ = MaxPooling2D(pool_size=(3, 3))(inputs_) 
    
    x = Conv2D(32,kernel_size=3,strides=1, padding='valid')(inputs)
    x = Activation('relu')(x)

    x = Conv2D(32,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Conv2D(64,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x) #Only one BN layer inbetween Convs (hoping for better transfer learning capabilities)

    x = Conv2D(64,kernel_size=3,strides=1, padding='valid')(x)
    x = Activation('relu')(x)

    x = Add()([x,inputs_])#Skip Connection

    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x) #Only one BN layer inbetween Dense (hoping for better transfer learning capabilities)

    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model

def VGG_small_4(in_dim,channels,out_dim): #Another change that allows input of different sizes
    inputs = Input(shape=(in_dim, in_dim,channels),name="inputTensor") #TensorFlow format; Keep and add to x after a convolution
    inputs_ = Conv2D(kernel_size=1, filters=64, strides=1, padding='same')(inputs)
    inputs_ = MaxPooling2D(pool_size=(4, 4))(inputs_) 
    
    x = Conv2D(32,kernel_size=3,strides=1, padding='same')(inputs)
    x = Activation('relu')(x)

    x = Conv2D(32,kernel_size=3,strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Conv2D(64,kernel_size=3,strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x) #Only one BN layer inbetween Convs (hoping for better transfer learning capabilities)
    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Conv2D(64,kernel_size=3,strides=1, padding='same')(x)
    x = Activation('relu')(x)

    x = Add()([x,inputs_])#Skip Connection

    x = MaxPooling2D(pool_size=(2, 2))(x) #Maxpool before dropout!

    x = Flatten()(x)  # this converts the 3D feature maps to 1D feature vectors

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x) #Only one BN layer inbetween Dense (hoping for better transfer learning capabilities)

    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])

    return model


def nitta_et_al_6layer(in_dim1,in_dim2,channels,out_dim):
    """
    The settins of this model are shown in the paper
    "Intelligent Image-Activated Cell Sorting" on Figure 5A
    """
    model = Sequential()
    
    model.add(Conv2D(32,3,3,input_shape=(in_dim1, in_dim2,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

    return model

def nitta_et_al_8layer(in_dim1,in_dim2,channels,out_dim):
    """
    The settins of this model are not shown in the paper
    I loaded the data of the paper and found the plt_model.h5, which 
    allows conclusions about model architecture (filter size, nr. of filters)
    I'm not sure about the dropout rates
    """
    model = Sequential()
    
    model.add(Conv2D(32,3,3,input_shape=(in_dim1, in_dim2,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('softmax',name="outputTensor"))

    return model



def Nitta_et_al_6layer_linact(in_dim1,in_dim2,channels,out_dim):
    """
    same as Nitta_et_al_6layer_linact, but with linear activation at the end
    """
    model = Sequential()
    
    model.add(Conv2D(32,3,3,input_shape=(in_dim1, in_dim2,channels),name="inputTensor")) #TensorFlow    
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('linear',name="outputTensor"))

    return model


#################Build models based on pretrainded weights#####################
def pretrained_xception(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (71,71,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.xception.Xception(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.xception.Xception(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
        
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_vgg16(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (48,48,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.vgg16.VGG16(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.vgg16.VGG16(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
        
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_vgg19(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (48,48,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.vgg19.VGG19(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.vgg19.VGG19(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_inception_v3(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (139,139,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.inception_v3.InceptionV3(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.inception_v3.InceptionV3(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = GlobalAveragePooling2D()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model



def pretrained_mobilenet(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Input must be of square shape of either (128, 128,3), (160, 160,3), (192, 192,3) or (224, 224,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.mobilenet.MobileNet(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.mobilenet.MobileNet(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
#    model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy'])
    return model


def pretrained_densenet(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (221,221,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.densenet.DenseNet121(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.densenet.DenseNet121(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model

def pretrained_nasnetlarge(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.nasnet.NASNetLarge(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.nasnet.NASNetLarge(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
        
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_nasnetmobile(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.nasnet.NASNetMobile(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.nasnet.NASNetMobile(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model



def pretrained_mobilenet_v2(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Input must be of square shape of either (96, 96,3), (128, 128,3), (160, 160,3), (192, 192,3) or (224, 224,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.mobilenet_v2.MobileNetV2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model

def pretrained_resnet50(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet50(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.resnet.ResNet50(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model

def pretrained_resnet101(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet101(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.resnet.ResNet101(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
    
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_resnet152(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet152(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    
    except:
        pretrained_model  = keras.applications.resnet.ResNet152(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
    
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model


def pretrained_resnet50_v2(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet50V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.resnet.ResNet50V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)

    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model

def pretrained_resnet101_v2(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet101V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.resnet.ResNet101V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
        
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model

def pretrained_resnet152_v2(in_dim1,in_dim2,channels,out_dim):
    """
    Loads a pretrained model from Keras (internet required).
    Minimum input dimension: (32,32,3)
    --Only three channels allowed!--
    It is recommended to use the expert model in AIDeveloper to only train the dense layers
    """
    try:
        pretrained_model  = keras_applications.resnet.ResNet152V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False,backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
    except:
        pretrained_model  = keras.applications.resnet.ResNet152V2(weights='imagenet',input_shape=(in_dim1,in_dim2,channels),include_top=False)
        
    layers = pretrained_model.layers
    layers[0].name = "inputTensor"
    #The output of the pretrained network are the inputs to our Dense layers
    x = Flatten()(pretrained_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    #combine pretrained models and our dense layers
    model = Model(input = pretrained_model.input, output = predictions) 
    return model



def collection_1(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    rand_state = np.random.RandomState(46) #to get the same random number on diff. PCs
    nodes_nr = [i*8 for i in range(1,9)]
    modelnames = []
    for i in nodes_nr:
        for j in nodes_nr:
            for k in nodes_nr:
                name = "MLP_"+str(i)+"_"+str(j)+"_"+str(k)
                modelnames.append(name)
    ind = rand_state.choice(range(len(modelnames)), size=100, replace=False)
    ind = np.sort(ind)
    modelnames_100 = np.array(modelnames)[ind]
    models_100 = []
    
    for modelname in modelnames_100:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models_100.append(model)
    return modelnames_100, models_100

def collection_2(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    rand_state = np.random.RandomState(46) #to get the same random number on diff. PCs
    nodes_nr = [i*8 for i in range(1,9)]
    modelnames = []
    for i in nodes_nr:
        for j in nodes_nr:
            for k in nodes_nr:
                name = "MLP_"+str(i)+"_"+str(j)+"_"+str(k)
                modelnames.append(name)
    ind = rand_state.choice(range(len(modelnames)), size=10, replace=False)
    ind = np.sort(ind)
    modelnames_100 = np.array(modelnames)[ind]
    models_100 = []
    
    for modelname in modelnames_100:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models_100.append(model)
    return modelnames_100, models_100


def collection_3(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    names1 = ["8,8,8","8,8,64","8,64,32"]
    names2 = ["16,16,16","16,64,40","16,24,64"]
    names2 = ["24,8,64","24,8,24","24,16,24","24,8,64"]
    names3 = ["32,8,8","32,8,64","32,16,8","32,24,16"]
    names4 = ["40,40,8","40,40,40","40,64,8","40,24,8"]
    names5 = ["48,40,8","48,40,40","48,64,8","48,24,8"]
    names6 = ["56,40,8","56,40,40","56,64,8","56,24,8"]
    names7 = ["64,40,8","64,48,16","64,64,64","64,24,8"]
    names = names1+names2+names3+names4+names5+names6+names7
    names = ["MLP_"+a.replace(",","_") for a in names]
    
    models = []
    for modelname in names:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models.append(model)
    return names, models


def collection_4(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    names1 = ["8,8,8","8,8,64","8,64,32"]
    names2 = ["16,16,16","16,64,40","16,24,64"]
    names = names1+names2
    names = ["MLP_"+a.replace(",","_") for a in names]
    
    models = []
    for modelname in names:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models.append(model)
    return names, models

def collection_5(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    rand_state = np.random.RandomState(46) #to get the same random number on diff. PCs
    nodes_nr = [i*4 for i in range(1,5)]
    modelnames = []
    for i in nodes_nr:
        for j in nodes_nr:
            for k in nodes_nr:
                name = "MLP_"+str(i)+"_"+str(j)+"_"+str(k)
                modelnames.append(name)
    #ind = rand_state.choice(range(len(modelnames)), size=100, replace=False)
    #ind = np.sort(ind)
    modelnames_100 = np.array(modelnames)#[ind]
    models_100 = []
    
    for modelname in modelnames_100:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models_100.append(model)
    return modelnames_100, models_100


def collection_6(in_dim1,in_dim2,channels,out_dim):
    """
    Return a collection of models and their names
    """
    rand_state = np.random.RandomState(46) #to get the same random number on diff. PCs
    nodes_nr = [i*8 for i in range(8,16)]
    modelnames = []
    for i in nodes_nr:
        for j in nodes_nr:
            for k in nodes_nr:
                name = "MLP_"+str(i)+"_"+str(j)+"_"+str(k)
                modelnames.append(name)
    ind = rand_state.choice(range(len(modelnames)), size=250, replace=False)
    ind = np.sort(ind)
    modelnames_100 = np.array(modelnames)#[ind]
    models_100 = []
    
    for modelname in modelnames_100:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname,in_dim1,channels,out_dim)
        models_100.append(model)
    return modelnames_100, models_100



def collection_test(in_dim1, in_dim2, channels, out_dim):
    """
    Return a small collection of models and their names
    """
    names = []
    for a in range(1, 3):
        for b in range(3):
                    n1 = a * 8
                    n2 = b * 8
                    m_str = str(n1)
                    if n2 > 0:
                        m_str += "," + str(n2)
                    ind = np.where(np.array(names) == m_str)[0]
                    if len(ind) < 1:
                        names.append(m_str)

    names = ["MLP_" + a.replace(",", "_") for a in names]

    models = []
    for modelname in names:
        modelname = modelname.split("MLP")[1]
        model = mlp_generator(modelname, in_dim1, channels, out_dim)
        models.append(model)
    return names, models


#########################Multi-Input models#######################


def MLP_MultiIn_3Lay_64(in_dim,channels,out_dim):
    inputA = Input(shape=(in_dim, in_dim,channels),name="inputTensorA") #TensorFlow format (height,width,channels)
    inputB = Input(shape=(1,),name="inputTensorB") 
    
    flattenA = Flatten()(inputA)
    #flattenB = Flatten()(inputB)
    x = Concatenate(axis=1)([flattenA, inputB])
    
    #x = Flatten()(x)

    x = Dense(64)(x)
    x = Dense(80)(x)
    x = Dense(32)(x)

    x = Activation('relu')(x)

    x = Dense(out_dim)(x)
    predictions = Activation('softmax',name="outputTensor")(x)
    
    model = Model(inputs=[inputA, inputB], outputs=predictions)

    return model

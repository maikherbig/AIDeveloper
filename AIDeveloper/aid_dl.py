# -*- coding: utf-8 -*-
"""
aid_dl
some useful functions deep learning
---------
@author: maikherbig
"""

import numpy as np
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import tensorflow as tf
from tensorflow.python.client import device_lib
device_types = device_lib.list_local_devices()
device_types = [device_types[i].device_type for i in range(len(device_types))]
config_gpu = tf.ConfigProto()
if device_types[0]=='GPU':
    config_gpu.gpu_options.allow_growth = True
    config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.7
from keras.models import clone_model
import keras_metrics #side package for precision, recall etc during training
global keras_metrics


def get_metrics_fresh(metrics,nr_classes):
    """
    Function takes a list of metrics and creates fresh tensors accordingly.
    This is necessary, after re-compiling the model because the placeholder has
    to be updated    
    """
    f1 = any(["f1_score" in str(a) for a in metrics])
    precision = any(["precision" in str(a) for a in metrics])
    recall = any(["recall" in str(a) for a in metrics])
    
    Metrics =  []
    if f1==True:
        for class_ in range(nr_classes):
            Metrics.append(keras_metrics.categorical_f1_score(label=class_))
    if precision==True:
        for class_ in range(nr_classes):
            Metrics.append(keras_metrics.categorical_precision(label=class_))
    if recall==True:
        for class_ in range(nr_classes):
            Metrics.append(keras_metrics.categorical_recall(label=class_))
    
    metrics =  ['accuracy'] + Metrics
    return metrics


def model_change_trainability(model_keras,trainable_new,model_metrics,out_dim):    
    #Function takes a keras model and list of trainable states.
    #The last n layers, which have parameters (not activation etc.) are set to 'not trainable'
    params = [model_keras.layers[i].count_params() for i in range(len(model_keras.layers))]
    #Only count layers that have parameters
    l_para_not_zero = np.where(np.array(params)>0)[0] #indices of layers that have parameters
    nr_layers = len(l_para_not_zero) #total nr. of layers of model with parameters
    #check that the number of layers in model is equal to the len(trainable_new) 
    assert nr_layers==len(trainable_new)
    #Original trainable states of layers with parameters
    trainable_now = [model_keras.layers[i].trainable for i in l_para_not_zero]
    if trainable_now == trainable_new:#they are already equal
        text = "No need to change trainable states of model."
    else:
        print("Change trainable states of model.")
        #Now change the trainability of each layer with parameters to the state given in trainable_states
        for i in range(nr_layers):
            layer_index = l_para_not_zero[i]
            model_keras.layers[layer_index].trainable = trainable_new[i]   
        #Model has to be recompiled in order to see any any effect of the change
        model_keras.compile(loss='categorical_crossentropy',
        optimizer='adam',metrics=get_metrics_fresh(model_metrics,out_dim))
    
        text = []
        model_keras.summary(print_fn=text.append)
        text = "\n".join(text)
    return text

def model_get_trainable_list(model_keras):
    params = [model_keras.layers[i].count_params() for i in range(len(model_keras.layers))]
    #Only count layers that have parameters
    l_para_not_zero = np.where(np.array(params)>0)[0] #indices of layers that have parameters
    #Original trainable states of layers with parameters
    trainable_list = [model_keras.layers[i].trainable for i in l_para_not_zero]
    layer_names_list = [model_keras.layers[i].__class__.__name__  for i in l_para_not_zero]
    return trainable_list,layer_names_list

def change_dropout(model_keras,do,model_metrics,out_dim):
    #Funktion takes a keras model and changes the dropout.
    #do = float (0 to 1) or list (with values from 0 to 1)
    #if do is a list: len(do) has to be equal to the nr. of dropout layers in the model
    #if do is a float: this value is then used for all droput layers
    model_keras._make_train_function()   
    model_weights = model_keras.get_weights() # Store weights
    optimizer_weights = model_keras.optimizer.get_weights() # Store optimizer values
    
    l_names = [model_keras.layers[i].__class__.__name__ for i in range(len(model_keras.layers))]
    ind = ["Dropout" in x for x in l_names]
    ind = np.where(np.array(ind)==True)[0] #at which indices are dropout layers?
    if len(ind)==0:#if there are no dropout layers, return
        return 0
    if type(do)==float:
        #set all dropout layers to do
        for i in range(len(ind)):
            model_keras.layers[ind[i]].rate = do
    elif type(do)==list:
        #check that the length of the list equals the nr. of dropout layers
        if len(do)!=len(ind):
            return 0
        else:
            for i in range(len(ind)):
                model_keras.layers[ind[i]].rate = do[i]
                
    #Only way that it actually has a effect is to clone the model, compile and reload weights
    model_keras = clone_model(model_keras) # If I do not clone, the new rate is never used. Weights are re-init now.
    model_keras.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=get_metrics_fresh(model_metrics,out_dim))
    
    model_keras._make_train_function()
    model_keras.set_weights(model_weights) # Set weights
    model_keras.optimizer.set_weights(optimizer_weights)#Set optimizer values
    return 1

def get_dropout(model_keras):
    #Funktion takes a keras model and gives a list of the dropout values.    
    l_names = [model_keras.layers[i].__class__.__name__ for i in range(len(model_keras.layers))]
    ind = ["Dropout" in x for x in l_names]
    ind = np.where(np.array(ind)==True)[0] #at which indices are dropout layers?
    do_list = [model_keras.layers[i].rate for i in ind]
    return do_list

def dump_to_simple_cpp(model_keras,model_config,output,verbose=False):
    """
    SOURCE:
        https://github.com/pplonski/keras2cpp
    Author: Piotr Plonski
    """
    with open(output, 'w') as fout:
        fout.write('layers ' + str(len(model_keras.layers)) + '\n')
    
        layers = []
        for ind, l in enumerate(model_config):
            if verbose:
                print(ind, l)
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
    
            if verbose:
                print(str(ind), l['class_name'])
            layers += [l['class_name']]
            if l['class_name'] == 'Conv2D':
                #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')
    
                #if 'batch_input_shape' in l['config']:
                #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
                #fout.write('\n')
    
                W = model_keras.layers[ind].get_weights()[0]
                if verbose:
                    print(W.shape)
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['padding'] + '\n')
    
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model_keras.layers[ind].get_weights()[1]) + '\n')
    
            if l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
            #if l['class_name'] == 'Flatten':
            #    print l['config']['name']
            if l['class_name'] == 'Dense':
                #fout.write(str(l['config']['output_dim']) + '\n')
                W = model_keras.layers[ind].get_weights()[0]
                if verbose:
                    print(W.shape)
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')

                for w in W:
                    fout.write(str(w) + '\n')
                fout.write(str(model_keras.layers[ind].get_weights()[1]) + '\n')


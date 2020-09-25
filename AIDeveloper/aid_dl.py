# -*- coding: utf-8 -*-
"""
aid_dl
functions to adjust and convert deep neural nets
grad_cam
LearningRateFinder
---------
@author: maikherbig
"""
import os, shutil,gc,tempfile
import numpy as np
import pandas as pd
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
import keras
from keras.models import load_model,Model
from keras.layers import Dense,Activation
from keras import backend as K

import keras_metrics #side package for precision, recall etc during training
global keras_metrics

from keras2onnx import convert_keras
from onnx import save_model as save_onnx
from mmdnn.conversion._script import convertToIR,IRToCode,convert
import coremltools
import cv2
import aid_bin

#Define some custom metrics which will allow to use precision, recall, etc during training
def get_custom_metrics():
    custom_metrics = []
    custom_metrics.append(keras_metrics.categorical_f1_score())
    custom_metrics.append(keras_metrics.categorical_precision())
    custom_metrics.append(keras_metrics.categorical_recall())
    custom_metrics = {m.__name__: m for m in custom_metrics}
    custom_metrics["sin"] = K.sin
    custom_metrics["abs"] = K.abs
    return custom_metrics


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


def model_change_trainability(model_keras,trainable_new,model_metrics,out_dim,loss_,optimizer_,learning_rate_):    
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
        model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim)
    
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

def get_optimizer_name(model_keras):
    optimizer = str(model_keras.optimizer)
    optimizer = optimizer.split("<keras.optimizers.")[1].split(" object at")[0]
    print(optimizer)
    if optimizer in ['SGD','RMSprop','Adagrad',"Adamax",'Adadelta','Adam','Nadam']:
        return optimizer
    else:
        return 0
    
def change_dropout(model_keras,do,model_metrics,out_dim,loss_,optimizer_,learning_rate_):
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
                
    #Only way that it actually has an effect is to clone the model, compile and reload weights
    model_keras = keras.models.clone_model(model_keras) # If I do not clone, the new rate is never used. Weights are re-init now.
    model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim)
    
    model_keras._make_train_function()
    model_keras.set_weights(model_weights) # Set weights
    model_keras.optimizer.set_weights(optimizer_weights)#Set optimizer values
    return 1

def model_add_classes(model_keras,nr_classes):
    #Sequential or Functional API?
    model_config = model_keras.get_config()
    if "sequential" not in model_config["name"]:
        print("Loaded model is defined using functional API of Keras")
        model_api = "functional"
    if "sequential" in model_config["name"]:
        print("Loaded model is defined using the sequential API of Keras")
        model_api = "sequential"
    
    #get list of layer types
    layer_types_list = [l.__class__.__name__ for l in model_keras.layers]
    #get the index of the last dense layer
    ind = [name=="Dense" for name in layer_types_list]#where are dense layers
    ind = np.where(np.array(ind)==True)[0][-1]#index of the last dense layer
    last_dense_name = model_keras.layers[ind].name #name of that last dense layer
    ind = ind-1 #go even one more layer back, because we need its output
    
    #For functional API:
    if model_api=="functional":
        #overwrite model_keras with a new, (shortened) model
        model_keras = Model(model_keras.get_input_at(0),model_keras.layers[ind].output)
        #Add a new final dense layer
        x = Dense(nr_classes,name=last_dense_name)(model_keras.layers[-1].output)
        x = Activation('softmax',name="outputTensor")(x)
        model_keras = Model(inputs=model_keras.get_input_at(0), outputs=x)
    #For sequential API:
    elif model_api=="sequential":
        model_keras.pop()#remove final activation layer
        model_keras.pop()#remove last dense layer
        model_keras.add(Dense(nr_classes,name=last_dense_name))
        model_keras.add(Activation('softmax',name="outputTensor"))
    
    #Compile to reset the optimizer weights
    model_keras.compile(optimizer='adam', loss='categorical_crossentropy')

    

def model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim):
    optimizer_name = optimizer_["comboBox_optimizer"].lower()
    if optimizer_name=='sgd':
        optimizer = keras.optimizers.SGD(lr=optimizer_["doubleSpinBox_lr_sgd"], momentum=optimizer_["doubleSpinBox_sgd_momentum"], nesterov=optimizer_["checkBox_sgd_nesterov"])
    elif optimizer_name=='rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=optimizer_["doubleSpinBox_lr_rmsprop"], rho=optimizer_["doubleSpinBox_rms_rho"])
    elif optimizer_name=='adagrad':
        optimizer = keras.optimizers.Adagrad(lr=optimizer_["doubleSpinBox_lr_adagrad"])
    elif optimizer_name=='adadelta':
        optimizer = keras.optimizers.Adadelta(lr=optimizer_["doubleSpinBox_lr_adadelta"], rho=optimizer_["doubleSpinBox_adadelta_rho"])
    elif optimizer_name=='adam':
        optimizer = keras.optimizers.Adam(lr=optimizer_["doubleSpinBox_lr_adam"], beta_1=optimizer_["doubleSpinBox_adam_beta1"], beta_2=optimizer_["doubleSpinBox_adam_beta2"], amsgrad=optimizer_["checkBox_adam_amsgrad"])
    elif optimizer_name=='adamax':
        optimizer = keras.optimizers.Adamax(lr=optimizer_["doubleSpinBox_lr_adamax"], beta_1=optimizer_["doubleSpinBox_adamax_beta1"], beta_2=optimizer_["doubleSpinBox_adamax_beta2"])
    elif optimizer_name=='nadam':
        optimizer = keras.optimizers.Nadam(lr=optimizer_["doubleSpinBox_lr_nadam"], beta_1=optimizer_["doubleSpinBox_nadam_beta1"], beta_2=optimizer_["doubleSpinBox_nadam_beta2"])
    else:
        print("Unknown optimizer!")
    model_keras.compile(loss=loss_,optimizer=optimizer,
                        metrics=get_metrics_fresh(model_metrics,out_dim))
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


def freeze_session(session, keep_var_names=None, input_names=None ,output_names=None, clear_devices=True):
    """
    Source:https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_names = input_names or []
        input_names += [v.op.name for v in tf.global_variables()]

        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph,input_names,output_names

def convert_kerastf_2_frozen_pb(model_path=None,outpath=None):
    """
    model_path = string; full path to the keras model (.model or .hdf)
    outfilename = string; full path and filename for the file to be created
    """
    if model_path==None:
        return 0
    elif os.path.isfile(model_path): #if is is a file (hopefully a keras file)
        tf.reset_default_graph() #Make sure to start with a fresh session
        sess = tf.InteractiveSession()
        model_keras = load_model(model_path,custom_objects=get_custom_metrics())#Load the model that should be converted          
        #Freeze the graph  
        frozen_graph,input_names,output_names = freeze_session(K.get_session(),input_names=[out.op.name for out in model_keras.inputs],output_names=[out.op.name for out in model_keras.outputs])
        #Get the names of the input and output node (required for optimization)
        out_path, out_fname = os.path.split(outpath)
        tf.train.write_graph(frozen_graph, out_path, out_fname, as_text=False)
        sess.close()
        return 1
        
def convert_kerastf_2_optimized_pb(model_path=None,outpath=None):
    """
    model_path = string; full path to the keras model (.model or .hdf)
    outfilename = string; full path and filename for the file to be created
    """
    if model_path==None or outpath==None:
        return 0
    elif not os.path.isfile(model_path): #if is is NOT a file (hopefully a keras file)
        return 0
    else: #Continue to the actual function
        tf.reset_default_graph() #Make sure to start with a fresh session
        sess = tf.InteractiveSession()
        model_keras = load_model(model_path,custom_objects=get_custom_metrics())#Load the model that should be converted          
        #Freeze the graph  
        frozen_graph,input_names,output_names = freeze_session(K.get_session(),input_names=[out.op.name for out in model_keras.inputs],output_names=[out.op.name for out in model_keras.outputs])
        #Get the names of the input and output node (required for optimization)
        input_names = [input_names[0]] #for now, only support single input models (one image)
        output_names = [output_names[0]] #for now, only support single output models (one prediction)
        
#        ##############More sophisticated finding of input/output nodes##########
#        #Find the nodes which are called 'inputTensor' 
#        ind = ["inputTensor" in s for s in input_names]
#        ind = np.where(np.array(ind)==True)[0]
#        if len(ind)==0:
#            print("there are no nodes with 'inputTensor'. Likely name='inputTensor' was not specified in model definition (in modelzoo.py)")
#            sess.close()
#            return 0 
#        elif len(ind)>1:
#            print("There are several nodes in models whose name contains 'inputTensor'. Please check if optimized model still works as expected")
#            input_names = list(np.array(input_names)[ind])
#        else:
#            input_names = list(np.array(input_names)[ind])
#
#        #Find the nodes which are called 'outputTensor' 
#        ind = ["outputTensor" in s for s in output_names]
#        ind = np.where(np.array(ind)==True)[0]
#        if len(ind)==0:
#            print("there are no nodes with 'outputTensor'. Likely name='outputTensor' was not specified in model definition (in modelzoo.py)")
#            sess.close()
#            return 0 
#        elif len(ind)>1:
#            print("There are several nodes in models whose name contains 'outputTensor'. Please check if optimized model still works as expected")
#            output_names = list(np.array(output_names)[ind])
#        else:
#            output_names = list(np.array(output_names)[ind])

        outputGraph = optimize_for_inference_lib.optimize_for_inference(frozen_graph,
                      input_names, # an array of the input node(s)
                      output_names, # an array of output nodes
        tf.int32.as_datatype_enum)
        # Save the optimized graph
        with tf.gfile.FastGFile(outpath, "w") as f:
            f.write(outputGraph.SerializeToString()) 
        sess.close()
        return 1

def convert_frozen_2_optimized_pb(model_path=None,outpath=None):
    """
    model_path = string; full path to the frozen model (.pb)
    outfilename = string; full path and filename for the file to be created
    """
    if model_path==None or outpath==None:
        return 0
    elif not os.path.isfile(model_path): #if is is NOT a file (hopefully a keras file)
        return 0
    else: #Continue to the actual function
        tf.reset_default_graph() #Make sure to start with a fresh session
        sess = tf.InteractiveSession()  
        #Load frozen .pb
        frozen_graph = tf.GraphDef()
        with tf.gfile.Open(model_path, "rb") as f:
          data2read = f.read()
          frozen_graph.ParseFromString(data2read)
        #Get the input_names and output_names
        frozen_nodes = list(frozen_graph.node)
        frozen_nodes_str = [node.name for node in frozen_nodes]
        ind = ["outputTensor" in s for s in frozen_nodes_str]
        ind = np.where(np.array(ind)==True)[0]
        if len(ind)==0:
            print("there are no nodes with the specific output name. Likely name='outputTensor' was not specified in model definition (in modelzoo.py)")
            sess.close()
            return 0 
        elif len(ind)>1:
            print("There are several nodes in models whose name contains 'outputTensor'. Please check if optimized model still works as expected")
            output_names = list(np.array(frozen_nodes_str)[ind])
        else:
            output_names = list(np.array(frozen_nodes_str)[ind])
        input_names = [frozen_nodes[0].name]
#        output_names = [frozen_nodes[-1].name]
        outputGraph = optimize_for_inference_lib.optimize_for_inference(frozen_graph,
                      input_names, # an array of the input node(s)
                      output_names, # an array of output nodes
        tf.int32.as_datatype_enum)
        # Save the optimized graph
        with tf.gfile.FastGFile(outpath, "w") as f:
            f.write(outputGraph.SerializeToString())
        sess.close()
        return 1

def convert_kerastf_2_onnx(model_path=None,outpath=None):
    tf.reset_default_graph() #Make sure to start with a fresh session
    sess = tf.InteractiveSession()
    model_keras = load_model(model_path,custom_objects=get_custom_metrics())#Load the model that should be converted          
    onnx_model = convert_keras(model_keras,model_keras.name)
    save_onnx(onnx_model, outpath)
    sess.close()

def convert_kerastf_2_script(model_path=None,out_format=None):
    """
    Convert model from keras.model to a script that allows to create the model
    using pytorch
    model_path - string, path to keras.model. model_path should point to a copy 
    of the model in the temp folder to make sure the relative path does not 
    contain spaces " ". Spaces cause isses due to a command-line interface of mmdnn.
    outpath
    out_format: string, indicating the target format. This can be "tensorflow","pytorch", "caffe","cntk","mxnet","onnx"
    """
    tf.reset_default_graph() #Make sure to start with a fresh session
    sess = tf.InteractiveSession()

    temp_path = aid_bin.create_temp_folder()#create a temp folder if it does not already exist
    #Create a  random filename for a temp. file
    tmp_model = np.random.choice(list("STERNBURGPILS"),5,replace=True)
    tmp_model = "".join(tmp_model)+".model"
    tmp_model = os.path.join(temp_path,tmp_model)
    shutil.copyfile(model_path,tmp_model) #copy the original model file there
    #Get the relative path to temp to make sure there are no spaces " " in the path
    relpath = os.path.relpath(tmp_model, start = os.curdir)
    
    #Keras to intermediate representation (IR): convert the temp. file
    parser = convertToIR._get_parser()
    dstPath = os.path.relpath(tmp_model, start = os.curdir)
    dstPath = os.path.splitext(dstPath)[0]#remove the .model file extension
    string = "--srcFramework keras --weights "+relpath+" --dstPath "+dstPath
    args = parser.parse_args(string.split())
    convertToIR._convert(args)
    
    #IR to Code
    dstModelPath = dstPath+".py"
    dstWeightPath = dstPath+".npy"
    IRModelPath = dstPath+".pb"
    parser = IRToCode._get_parser()
    string = "--dstFramework "+out_format+" --dstModelPath "+dstModelPath+" --IRModelPath "+IRModelPath+" --IRWeightPath "+dstWeightPath+" --dstWeightPath "+dstWeightPath
    args = parser.parse_args(string.split())
    IRToCode._convert(args)
    
    #Copy the final output script and weights back to the original folder
    out_script = os.path.splitext(model_path)[0]+".py" #remove .model, put .py instead
    out_weights = os.path.splitext(model_path)[0]+".npy" #remove .model, put .py instead
    shutil.copyfile(dstModelPath,out_script) #copy from temp to original path
    shutil.copyfile(dstWeightPath,out_weights) #copy from temp to original path
    
    #delete all the temp. files
    del_json = os.path.splitext(relpath)[0]+".json"
    for file in [IRModelPath,dstModelPath,dstWeightPath,relpath,del_json]:
        try:
            os.remove(file)
            print("temp. file deleted: "+file)
        except:
            print("temp. file not found/ not deleted: "+file)
    sess.close()

def convert_kerastf_2_onnx_mmdnn(model_path):
    tf.reset_default_graph() #Make sure to start with a fresh session
    sess = tf.InteractiveSession()

    temp_path = aid_bin.create_temp_folder()#create a temp folder if it does not already exist
    #Create a  random filename for a temp. file
    tmp_model = np.random.choice(list("STERNBURGPILS"),5,replace=True)
    tmp_model = "".join(tmp_model)+".model"
    tmp_model = os.path.join(temp_path,tmp_model)
    shutil.copyfile(model_path,tmp_model) #copy the original model file there
    #Get the relative path to temp to make sure there are no spaces " " in the path
    relpath = os.path.relpath(tmp_model, start = os.curdir)
    tmp_out_model = os.path.splitext(relpath)[0]+".onnx" 
    string = "--srcFramework keras --inputWeight "+relpath+" --dstFramework onnx --outputModel "+tmp_out_model
    parser = convert._get_parser()
    args, unknown_args = parser.parse_known_args(string.split())
    temp_filename = os.path.splitext(relpath)[0]+"conv"
    convert._convert(args,unknown_args,temp_filename)

    out_model_onnx = os.path.splitext(model_path)[0]+".onnx"
    shutil.copyfile(tmp_out_model,out_model_onnx) #copy the original model file there

    #delete all the temp. files
    for file in [tmp_out_model,tmp_model]:
        try:
            os.remove(file)
            print("temp. file deleted: "+file)
        except:
            print("temp. file not found/ not deleted: "+file)
            
    sess.close()

def convert_kerastf_2_coreml(model_path):
    tf.reset_default_graph() #Make sure to start with a fresh session
    sess = tf.InteractiveSession()
    path_out = os.path.splitext(model_path)[0] + ".mlmodel"
    model = coremltools.converters.keras.convert(model_path, input_names=['inputTensor'],output_names=['outputTensor'],model_precision='float32',use_float_arraytype=True,predicted_probabilities_output="outputTensor")
    model.save(path_out)
    sess.close()


def get_config(cpu_nr,gpu_nr,deviceSelected,gpu_memory):
    #No GPU available, CPU selected:
    if gpu_nr==0: #and deviceSelected=="Default CPU":
        print("Adjusted options for CPU usage")
        config_gpu = tf.ConfigProto()
    #GPU available but user wants to use CPU
    elif gpu_nr>0 and deviceSelected=="Default CPU":
        config_gpu = tf.ConfigProto(intra_op_parallelism_threads=cpu_nr,\
                inter_op_parallelism_threads=cpu_nr, allow_soft_placement=True,\
                device_count = {'CPU' : 1, 'GPU' : 0})
        print("Adjusted options for CPU usage")
    
    #GPU selected
    elif deviceSelected=="Single-GPU" or deviceSelected=="Multi-GPU":
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory,\
                                    allow_growth = True)
        config_gpu = tf.ConfigProto(allow_soft_placement=True,\
                                    gpu_options = gpu_options,\
                                    log_device_placement=True)
        if deviceSelected=="Single-GPU":
            print("Adjusted GPU options for Single-GPU usage. Set memeory fraction to "+str(gpu_memory))
        if deviceSelected=="Multi-GPU":
            print("Adjusted GPU options for Multi-GPU usage. Set memeory fraction to "+str(gpu_memory))
    
#        if deviceSelected=="Multi-GPU":
#            for device in gpu_devices:
#                tf.config.experimental.set_memory_growth(device, True)
#            config_gpu.gpu_options.per_process_gpu_memory_fraction = gpu_memory

    return config_gpu

def reset_keras(model=None,create_new_config=False):
    "Source: https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/18"
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    garbage = gc.collect() # if it's done something you should see a number being outputted
    
    if create_new_config:
        # use the same config as you used to create the session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        K.set_session(tf.Session(config=config))


def get_last_conv_layer_name(model_keras):
    """
    Search for the last convolutional layer
    Args:
        model_keras: A keras model object

    Returns:
        Name of the layer (str)
    """
    for layer in reversed(model_keras.layers):#loop in reverse order
        # Select closest 4D layer to the end of the network.
        if len(layer.output_shape) == 4:
            return layer.name

    raise ValueError("Could not find a convolutional layer (layer with 4D).")


def grad_cam(load_model_path, images, class_, layer_name):
    """
    Reference to paper:
    -https://arxiv.org/abs/1610.02391
    
    Args:
        model_keras: A keras model object
        image: N images as numpy array. Dimension: [N,W,H,C]
        class_: Integer (int) indicating, for which class Grad-CAM should be computed
        layer_name: Name of the last convolutional layer
    Returns:
       List: List of numpy arrays (Grad-CAM heatmaps), each of the same size as the input images
    """
    tf.reset_default_graph() #Make sure to start with a fresh session
    sess = tf.InteractiveSession()
    
    model_keras = load_model(load_model_path,custom_objects=get_custom_metrics())  
    
    #get loss for class_
    class_output = model_keras.output[:, class_]
    #get output of the last convolutional layer
    convolution_output = model_keras.get_layer(layer_name).output
    #compute gradients
    grads = K.gradients(class_output, convolution_output)[0]
    #define a function mapping from the oringinal input (image) to the values of the last conv. layer and the gradients
    gradient_function = K.function([model_keras.input], [convolution_output, grads])

    conv_out, grads_out = gradient_function([images])

    #averaging all final feature maps
    weights = np.mean(grads_out, axis=(1,2))
    cam = [np.dot(conv_out[i], weights[i]) for i in range(conv_out.shape[0])]
    #normalize between 0 and 255
    cam = [cam_ -np.min(cam_) for cam_ in cam] #subtract minimum 
    cam = [(cam_/np.max(cam_))*255.0 for cam_ in cam] #divide by maximum; multiply with 255
    cam = [cam_.astype(np.uint8) for cam_ in cam]#convert to uint8

    #resize cam to get a heatmap of the same size as input image; 
    #Take note that resize happens after normalization since OpenCV is much faster with uint8 images
    cam = [cv2.resize(cam_, (images[0].shape[0], images[0].shape[1]), cv2.INTER_LINEAR) for cam_ in cam]

    sess.close()
    
    return cam


class LearningRateFinder:
    """
    Citation: Rosebrock, A (2019) Keras Learning Rate Finder [Source code].
    https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder
    """
    def __init__(self, model, stopFactor=4, beta=0.98):
		# store the model, stop factor, and beta value (for computing
		# a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = [] #list for learning rates
        self.losses_sm,self.losses_or,self.val_losses_sm,self.val_losses_or = [],[],[],[]#list for losses and val.losses
        self.accs_sm,self.accs_or, self.val_accs_or,self.val_accs_sm = [],[],[],[]#list for accuracies
        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.avgLossV = 0
        self.avgAcc = 0
        self.avgAccV = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = [] #list for learning rates
        self.losses_sm,self.losses_or,self.val_losses_sm,self.val_losses_or = [],[],[],[]#list for losses and val.losses
        self.accs_sm,self.accs_or, self.val_accs_or,self.val_accs_sm = [],[],[],[]#list for accuracies
        self.lrMult = 1
        self.avgLoss = 0
        self.avgAcc = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
	
    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
        "DataFrameIterator", "Iterator", "Sequence"]
        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value

#        print("logs")
#        print(logs)
        self.batchNum += 1
        
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * logs["loss"])
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses_sm.append(smooth)
        self.losses_or.append(logs["loss"])

        self.avgAcc = (self.beta * self.avgAcc) + ((1 - self.beta) * logs["acc"])
        smooth = self.avgAcc / (1 - (self.beta ** self.batchNum))
        self.accs_sm.append(smooth)
        self.accs_or.append(logs["acc"])

        #print(logs["acc"],smooth,self.beta,self.batchNum)


        #Get validation accuracy
        if self.valData!=None:
            val_loss, val_acc = self.model.evaluate(self.valData[0], self.valData[1], verbose=0)
            
            self.avgLossV = (self.beta * self.avgLossV) + ((1 - self.beta) * val_loss)
            smooth = self.avgLossV / (1 - (self.beta ** self.batchNum))
            self.val_losses_sm.append(smooth)
            self.val_losses_or.append(val_loss)

            self.avgAccV = (self.beta * self.avgAccV) + ((1 - self.beta) * val_acc)
            smooth = self.avgAccV / (1 - (self.beta ** self.batchNum))
            self.val_accs_sm.append(smooth)
            self.val_accs_or.append(val_acc)


        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss
        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return
        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)


    def find(self, trainData,valData, startLR, endLR, epochs=None,
        stepsPerEpoch=None, batchSize=32, sampleSize=2048,verbose=1):
        
        self.valData = valData #put validation data on self to allow on_batch_end to compute val_acc
        
        self.reset()# reset our class-specific variables
        # determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)
        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
        # if we're not using a generator then our entire dataset must
        # already be in memory
        elif not useGen:
            # grab the number of samples in the training data and
            # then derive the number of steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))
        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))
            
        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch
        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)            
                    
        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        # check to see if we are using a data iterator
        if useGen:
            self.model.fit(
            x=trainData,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback])
        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                x=trainData[0], y=trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback],
                verbose=verbose)
        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)            
        

class cyclicLR(keras.callbacks.Callback):
    """
    Reference: https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    Author: Brad Kenstler
    
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = cyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = cyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(cyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

class exponentialDecay(keras.callbacks.Callback):
    """    
    Decrease learning rate using exponential decay
    lr = initial_lr * decay_rate ** (batch_iteration / decay_steps)    
    """
    def __init__(self, initial_lr=0.01, decay_steps=100, decay_rate=0.95):
        super(exponentialDecay, self).__init__()

        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_initial_lr=None, new_decay_steps=None,new_decay_rate=None):
        if new_initial_lr != None:
            self.initial_lr = new_initial_lr
        if new_decay_steps != None:
            self.decay_steps = new_decay_steps
        if new_decay_rate != None:
            self.decay_rate = new_decay_rate
        self.iterations = 0.
        
    def exp_decay(self):
        return self.initial_lr * self.decay_rate ** (self.iterations / self.decay_steps)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.iterations == 0:
            K.set_value(self.model.optimizer.lr, self.initial_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.exp_decay())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.exp_decay())



def get_cyclStepSize(SelectedFiles,step_size,batch_size):
    ind = [selectedfile["TrainOrValid"] == "Train" for selectedfile in SelectedFiles]
    ind = np.where(np.array(ind)==True)[0]
    SelectedFiles_train = np.array(SelectedFiles)[ind]
    SelectedFiles_train = list(SelectedFiles_train)
    nr_events_train_total = np.sum([int(selectedfile["nr_events_epoch"]) for selectedfile in SelectedFiles_train])

    cycLrStepSize = step_size*int(np.round(nr_events_train_total / batch_size))#number of steps in one epoch 
    return cycLrStepSize           


def get_lr_dict(learning_rate_const_on,learning_rate_const,
                    learning_rate_cycLR_on,cycLrMin,cycLrMax,
                    cycLrMethod,cycLrStepSize,
                    learning_rate_expo_on,
                    expDecInitLr,expDecSteps,expDecRate,cycLrGamma):
    lr_dict = pd.DataFrame()
    lr_dict["learning_rate_const_on"] = learning_rate_const_on,
    lr_dict["learning_rate_const"] = learning_rate_const,
    lr_dict["learning_rate_cycLR_on"] = learning_rate_cycLR_on,
    lr_dict["cycLrMin"] = cycLrMin,
    lr_dict["cycLrMax"] = cycLrMax,
    lr_dict["cycLrMethod"] = cycLrMethod,
    lr_dict["cycLrStepSize"] = cycLrStepSize,
    lr_dict["expDecInitLr"] = expDecInitLr,
    lr_dict["expDecInitLr"] = expDecInitLr,
    lr_dict["expDecSteps"] = expDecSteps,
    lr_dict["expDecRate"] = expDecRate,
    lr_dict["cycLrGamma"] = cycLrGamma,
    return lr_dict

def get_optimizer_settings():
    """
    Default optimizer settings (taken from https://keras.io/api/optimizers)
    """
    d = {}
    d["comboBox_optimizer"] = "Adam"

    d["doubleSpinBox_lr_sgd"] = 0.01
    d["doubleSpinBox_sgd_momentum"] = 0.0
    d["checkBox_sgd_nesterov"] = False

    d["doubleSpinBox_lr_rmsprop"] = 0.001
    d["doubleSpinBox_rms_rho"] = 0.9

    d["doubleSpinBox_lr_adam"] = 0.001
    d["doubleSpinBox_adam_beta1"] = 0.9
    d["doubleSpinBox_adam_beta2"] = 0.999
    d["checkBox_adam_amsgrad"] = False

    d["doubleSpinBox_lr_nadam"] = 0.002
    d["doubleSpinBox_nadam_beta1"] = 0.9
    d["doubleSpinBox_nadam_beta2"] = 0.999

    d["doubleSpinBox_lr_adadelta"] = 1.0
    d["doubleSpinBox_adadelta_rho"] = 0.95

    d["doubleSpinBox_lr_adagrad"] = 0.01

    d["doubleSpinBox_lr_adamax"] = 0.002
    d["doubleSpinBox_adamax_beta1"] = 0.9
    d["doubleSpinBox_adamax_beta2"] = 0.999

    return d


def get_lr_callback(learning_rate_const_on,learning_rate_const,
                    learning_rate_cycLR_on,cycLrMin,cycLrMax,
                    cycLrMethod,cycLrStepSize,
                    learning_rate_expo_on,
                    expDecInitLr,expDecSteps,expDecRate,cycLrGamma):

    if learning_rate_const_on==True:
        return None
    elif learning_rate_cycLR_on==True:
        return cyclicLR(base_lr=cycLrMin,max_lr=cycLrMax,step_size=cycLrStepSize,mode=cycLrMethod,gamma=cycLrGamma)  
    elif learning_rate_expo_on==True:
        return exponentialDecay(initial_lr=expDecInitLr, decay_steps=expDecSteps, decay_rate=expDecRate)  

       

        
        
        
        
        
        
        

        

    

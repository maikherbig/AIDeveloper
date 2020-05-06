# -*- coding: utf-8 -*-
"""
aid_dl
functions adjust and convert deep neural nets
---------
@author: maikherbig
"""
import os, shutil,gc
import numpy as np
rand_state = np.random.RandomState(117) #to get the same random number on diff. PCs
import tensorflow as tf
#from tensorflow.python.client import device_lib
#device_types = device_lib.list_local_devices()
#device_types = [device_types[i].device_type for i in range(len(device_types))]
#config_gpu = tf.ConfigProto()
#if device_types[0]=='GPU':
#    config_gpu.gpu_options.allow_growth = True
#    config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.7
import keras
import keras_metrics #side package for precision, recall etc during training
global keras_metrics

from tensorflow.python.tools import optimize_for_inference_lib
from keras.models import load_model
from keras import backend as K
from keras2onnx import convert_keras
from onnx import save_model as save_onnx
from mmdnn.conversion._script import convertToIR,IRToCode,convert
import coremltools
import aid_bin

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
    if optimizer in ['SGD','RMSprop','Adagrad','Adadelta','Adam','Nadam']:
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

def model_compile(model_keras,loss_,optimizer_,learning_rate_,model_metrics,out_dim):
    optimizer_ = optimizer_.lower()
    if optimizer_=='sgd':
        optimizer_ = keras.optimizers.SGD(lr=learning_rate_, momentum=0.0, nesterov=False)
    elif optimizer_=='rmsprop':
        optimizer_ = keras.optimizers.RMSprop(lr=learning_rate_, rho=0.9)
    elif optimizer_=='adagrad':
        optimizer_ = keras.optimizers.Adagrad(lr=learning_rate_)
    elif optimizer_=='adadelta':
        optimizer_ = keras.optimizers.Adadelta(lr=learning_rate_, rho=0.95)
    elif optimizer_=='adam':
        optimizer_ = keras.optimizers.Adam(lr=learning_rate_, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer_=='adamax':
        optimizer_ = keras.optimizers.Adamax(lr=learning_rate_, beta_1=0.9, beta_2=0.999)
    elif optimizer_=='nadam':
        optimizer_ = keras.optimizers.Nadam(lr=learning_rate_, beta_1=0.9, beta_2=0.999)
    else:
        print("Unknown optimizer!")
    model_keras.compile(loss=loss_,optimizer=optimizer_,
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
        model_keras = load_model(model_path)#Load the model that should be converted          
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
        model_keras = load_model(model_path)#Load the model that should be converted          
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
    model_keras = load_model(model_path)#Load the model that should be converted          
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
        

    

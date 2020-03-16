import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from keras.models import load_model
from keras import backend as K
from keras2onnx import convert_keras
from onnx import save_model as save_onnx

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














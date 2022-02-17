"""
Library to create neural nets with partially trainable layers.
In Keras currently, only the trainablilty of whole layers can be set.
To create partial trainabliity, I need to split the layer into two layers,
and assign one to trainable and the other one to not trainable. Both layers get
the same input. The output of both layers is then concatenated. The resulting 
neural net performs identically to the original neural net
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import copy
import model_zoo


def partial_trainability(model_keras,Layer_names,Layer_trainabliities):
    """
    This function allows you to get partially trainable layers
    function takes
    model_keras: keras model you have loaded or created for example using model_zoo.get_model("VGG_small_4",32,3,2)
    Layer_names: list containing the names of the layers that should be changed. Of course these names have to be existing in model_keras
    Layer_trainability: list containing floats. For each layer that should be changed, provide a value between 0 and 1 (0-layer not trainable at all; 1-layer entirely trainable)
    
    Lets say you use Layer_names=['dense_3'] and Layer_trainability=[0.25]
    Lets assume 'dense_3' has 1000 nodes.
    Then two new parallell dense layers will be created. 
    The first one gets 750 nodes, which are set to set to Trainable=False.
    The second dense layer has 250 nodes, which are trainable.
    The corresponding weights from the initial model are distributed to these 
    layer accordingly.    
    """
    #Get the config of the current model
    model_config = model_keras.get_config()
    #Deep-Copy the config of the  original model
    model_config_new = copy.deepcopy(model_config)
    
    if type(model_config_new)==dict:
        print("Model used functional API of Keras")
    elif type(model_config_new)==list:
        print("Model used sequential API of Keras and will now be converted to functional API")
        #api = "Sequential"
        #Convert to functional API        
        input_layer = Input(batch_shape=model_keras.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model_keras.layers:
            prev_layer = layer(prev_layer)
        model_keras = Model([input_layer], [prev_layer]) 
        #Now we have functional API :)
        #Get the model config for the converted model
        model_config = model_keras.get_config()
        model_config_new = copy.deepcopy(model_config)
    else:
        print("Unknown format for model config")
        #return
        
    Layer_nontrainability = list(1-np.array(Layer_trainabliities))
    del Layer_trainabliities
    #Original names of the layers
    Layer_names_orig = [model_keras.layers[i].name for i in range(len(model_keras.layers))]
    #Now we are starting to loop over all the requested changes:
    for i in range(len(Layer_names)):
        nontrainability = Layer_nontrainability[i]
        layer_name = Layer_names[i]
        layer_names_list = [model_config_new["layers"][it]['name'] for it in range(len(model_config_new["layers"]))]
        layer_index = int(np.where(np.array(layer_names_list)==layer_name)[0])
        layer_config = model_config_new['layers'][layer_index]
        layer_type = layer_config["class_name"]
        if layer_type == "Dense":
            split_property = "units" #'units' are the number of nodes in dense layers
        elif layer_type == "Conv2D":
            split_property = "filters"
        
        nr_nodes = layer_config["config"][split_property]
        nr_const = int(np.round(nontrainability*nr_nodes))
        #get a config for the non-trainable part
        
        layer_config_const = copy.deepcopy(layer_config)
        layer_name_const = layer_config_const['config']["name"]+"_1"
        layer_config_const["config"][split_property] = nr_const
        layer_config_const["config"]["trainable"] = False #not trainable
        layer_config_const["config"]["name"] = layer_name_const#rename 
        layer_config_const["name"] = layer_name_const#rename 
        
        #get a config for the rest (trainable part)
        layer_config_rest = copy.deepcopy(layer_config)
        #this part will only exist if nr_nodes-nr_const>0:
        if nr_nodes-nr_const>0:
            layer_name_rest = layer_config_rest["config"]["name"]+"_2"
            layer_config_rest["config"][split_property] = nr_nodes-nr_const
            layer_config_rest["config"]["name"] = layer_name_rest#rename 
            layer_config_rest["name"] = layer_name_rest#rename 
        
        #Assemble a config for a corresponding concatenate layer
        inbound_1 = [layer_name_const,0,0,{}]
        inbound_2 = [layer_name_rest,0,0,{}]
        inbound_nodes = [inbound_1,inbound_2]
        layer_name = layer_config['config']["name"]#Call it like the initial layer that has been there. This will allow other layers to correctly conncet like before #'concatenate_'+layer_config["config"]["name"]
        config_conc = {"axis":-1,"name":layer_name,"trainable":True}
        conc_dict = {"class_name":'Concatenate',"config":config_conc,"inbound_nodes":[inbound_nodes],"name":layer_name}
        
        #insert these layers into the config at Index: layer_index
        layerlist = model_config_new["layers"]
        #Replace existing layer with the const part
        layerlist[layer_index] = layer_config_const
        #After that insert the rest part
        layerlist.insert(layer_index+1, layer_config_rest)
        #After that insert the concatenate layer
        layerlist.insert(layer_index+2, conc_dict)
        #Update the config with the new layers
        model_config_new["layers"] = layerlist    
    
    #Build the model using this updated config
    model_keras_new = Model.from_config(model_config_new)
    #Compilation might not be required.
    model_keras_new.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    iterator = 0   
    for layer_name in Layer_names_orig: #for all layers of the original model
        layer = model_keras.get_layer(layer_name)
        if layer_name not in Layer_names: #if this layer was not subjected to change in the new model...
            #Straight forward: get weights of the orignal model
            weights = layer.get_weights()
            #And put them into the new model
            model_keras_new.get_layer(layer_name).set_weights(weights)
        else:#Otherwise the layer was changed        
            layer_type = layer.__class__.__name__
            layer_config = layer.get_config()
            weights = layer.get_weights()#Get the original weights
            trainability = Layer_nontrainability[iterator]
            iterator += 1
            
            if layer_type == "Dense":
                split_property = "units" #'units' are the number of nodes in dense layers
            elif layer_type == "Conv2D":
                split_property = "filters"
           
            nr_nodes = layer_config[split_property]
            nr_const = int(np.round(trainability*nr_nodes))
            
            #Constant part
            if layer_type == "Dense":
                #Put the former weights into the layer
                weights_const = list(np.copy(weights))
                weights_const[0] = weights_const[0][:,0:nr_const]
                weights_const[1] = weights_const[1][0:nr_const]
            elif layer_type == "Conv2D":
                #Put the former weights into the layer
                weights_const = list(np.copy(weights))
                weights_const[0] = weights_const[0][:,:,:,0:nr_const]
                weights_const[1] = weights_const[1][0:nr_const]
            else:
                print("Unknown layer type")
                #return
            layer_name_const = layer_name+"_1"#rename         
            model_keras_new.get_layer(layer_name_const).set_weights(weights_const)
    
            #Rest trainable part
            #this part will only exist if nr_nodes-nr_const>0:
            if nr_nodes-nr_const>0:            
                if layer_type == "Dense":
                    #Get the weights
                    weights_rest = list(np.copy(weights))
                    weights_rest[0] = weights_rest[0][:,nr_const:]
                    weights_rest[1] = weights_rest[1][nr_const:]
                if layer_type == "Conv2D":
                    #Get the weights
                    weights_rest = list(np.copy(weights))
                    weights_rest[0] = weights_rest[0][:,:,:,nr_const:]
                    weights_rest[1] = weights_rest[1][nr_const:]
                       
                layer_name_rest = layer_name+"_2"#rename         
                model_keras_new.get_layer(layer_name_rest).set_weights(weights_rest)
    return model_keras_new

def test_partial_trainability():
    #Define some model to start with
    model_keras = model_zoo.get_model("VGG_small_4",32,3,2)
    #model_keras = model_zoo.get_model("LeNet5_do",32,3,2)

#    model_keras1_c = model_keras1.get_config()
#    model_keras2_c = model_keras2.get_config()

    
    Layer_types_orig = [model_keras.layers[i].__class__.__name__ for i in range(len(model_keras.layers))]
    Layer_names_orig = [model_keras.layers[i].name for i in range(len(model_keras.layers))]    
    
    #Count Dense and Conv layers
    is_dense_or_conv = [layer_type in ["Dense","Conv2D"] for layer_type in Layer_types_orig] 
    index = np.where(np.array(is_dense_or_conv)==True)[0]
    #That would be the stuff, written in the table in AID
    Layer_names = np.array(Layer_names_orig)[index]
    
    #Provide a list with layer names, which should be changed and the corresponding trainabilities
    Layer_names = [Layer_names[0],Layer_names[1],Layer_names[2]]
    Layer_trainabliities = [0.2,0.4,0.6]
    
    model_keras_new = partial_trainability(model_keras,Layer_names,Layer_trainabliities)

    shape = list(model_keras_new.layers[0].input_shape)
    shape[0] = 1
    img_rndm = np.random.randint(low=0,high=255,size=shape)
    img_rndm = img_rndm.astype(float)/255.0
    
    #Both models should perform identically
    p1 = model_keras.predict(img_rndm)
    p2 = model_keras_new.predict(img_rndm)
    assert np.allclose(p1, p2)
    
    #Also start a fititng processs
    shape_tr = shape
    shape_tr[0] = 250
    train_x = np.random.randint(low=0,high=255,size=shape)
    train_x = train_x.astype(float)/255.0
    train_y = np.r_[np.repeat(0,125),np.repeat(1,125)]
    train_y_ = to_categorical(train_y, 2)# * 2 - 1
    model_keras_new.fit(train_x,train_y_,epochs=1)
























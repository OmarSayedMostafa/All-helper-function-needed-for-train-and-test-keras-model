from keras.models import load_model
import time
import json
import numpy as np
from keras.layers import Input
import ast

calling_sequence = []


def write_json_parameters(parameters):
    # Writing JSON file
    parameters['optimizer']='adam'
    for key in parameters:
        parameters[key] = str(parameters[key])
    with open(parameters['json_file_path']+'/parameters.json', 'w') as json_file:
        json.dump(parameters, json_file)
        


def load_json_parameters(filepath):
    #Open JSON file
    parameters = None
    with open(filepath) as f:
        parameters = json.load(f)
    
    # Convet String to Values, Lists
    parameters["total_epochs"] = ast.literal_eval(parameters["total_epochs"])

    parameters["input_height"] = ast.literal_eval(parameters["input_height"])

    parameters["input_width"] = ast.literal_eval(parameters["input_width"])

    parameters["input_channels"] = ast.literal_eval(parameters["input_channels"])

    parameters["batch_size"] = ast.literal_eval(parameters["batch_size"] )

    parameters["drop_rate"] = ast.literal_eval(parameters["drop_rate"])

    parameters["shift_range"] = ast.literal_eval(parameters["shift_range"])

    parameters["rotation_range"] = ast.literal_eval(parameters["rotation_range"])
    
    parameters["top_layers_dims"] = ast.literal_eval(parameters["top_layers_dims"])


    parameters["classes"] = ast.literal_eval(parameters["classes"])

    parameters["train_classes_names"] = ast.literal_eval(parameters["train_classes_names"])

    parameters["train_data_per_class"] = ast.literal_eval(parameters["train_data_per_class"])

    parameters["valid_classes_names"] = ast.literal_eval(parameters["valid_classes_names"])

    parameters["valid_data_per_class"] = ast.literal_eval(parameters["valid_data_per_class"])

    parameters["test_classes_names"] = ast.literal_eval(parameters["test_classes_names"])

    parameters["test_data_per_class"] = ast.literal_eval(parameters["test_data_per_class"])
    
    return parameters









def save_parameters(parameters):
    write_json_parameters(parameters)



def load_parameters(json_file_path):
    return load_json_parameters(json_file_path)



def load_pretrained_model(model_path=None):
    start_loading = time.time()
    model = load_model(model_path)
    load_time = time.time()-start_loading
    print("==>> model loaded in "+str(load_time)+"sec")
    return model



# create tensor shape for model 
def create_input_tensor_shape(parameters):
    # append function name to the call sequence
    calling_sequence.append("[create_input_tensor_shape]==>>")
    print(" ==============================================")
    print(" [INFO] Entering function[create_input_tensor_shape]  in core.py")
    print(" [INFO] data formate = ", parameters['dataformat'])

    input_tensor_shape = None
    if parameters['dataformat'] == 'channels_last': # creat input tensor with shape [h, w, c]
        input_tensor_shape = Input(shape=(parameters['input_height'], parameters['input_width'],
                                    parameters['input_channels'])) 
    else: # create input tensor with shape = [c, h, w]
        input_tensor_shape = Input(shape=(parameters['input_channels'],parameters['input_height'],
                                    parameters['input_width']))

    print(" [INFO] Leaving function[create_input_tensor_shape]  in core.py with tensor shape="+str(input_tensor_shape))
    print(" ==============================================")
    return input_tensor_shape


def get_classes_names_and_data_count_per_class_from_generator(generator):
    # append function name to the call sequence
    calling_sequence.append("[get_classes_names_and_data_count_per_class_from_generator]==>>")
    
    print(" ==============================================")    
    print(" [INFO] Entering function[get_classes_names_and_data_count_per_class_from_generator]  in core.py")
    #-------------------------------------------------------------------------        
    # return data characterstic 
    # classes count
    # classes names
    # data size for each class    
    classes = generator.classes # array[0,0,0,...,1,1,1,....,2,2,2,..]# get all data classes ids 
    class_names = list(generator.class_indices.keys()) # get classes names/keys from classes dictionary contains the names and the ids
    classes, data_count = np.unique(classes, return_counts=True) # get the count of unique values in the data list 
    # get data size according to unique class  
    data_per_class = list(dict(zip(classes, data_count)).values())
    #-------------------------------------------------------------------------
    print(" [INFO] Leaaving function[get_classes_names_and_data_count_per_class_from_generator]  in core.py with classes names ="+str(class_names)+" \n and data per class ="+str(data_per_class))
    print(" ==============================================")
    return class_names, data_per_class
  
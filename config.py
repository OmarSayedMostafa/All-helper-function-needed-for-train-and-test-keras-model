from utils import calling_sequence, np, write_json_parameters
import sys

parent_directory='/full/path/to/your/parent/directory'
model_save_path = '/path/to/your/weights'
train_data_path = '/path/to/your/train_data/'
valid_data_path = 'path/to/your/valid_data/'
test_data_path = '/path/to/your/test_data/'

result_sheet = '/path/to/your/result_sheet.csv'

parameters_json_file_path = model_save_path


parameters = {
    'model_name': '',
    'Dataset_name': '',
    
    'train_acc_according_to_max_validation_acc': 0.0,
    'max_validation_acc' : 0.0,
    
    'best_accuracy_epoch' : 0,
    'total_epochs' : 50,
    
    'overall_train_accuracy':0.0,
    'overall_validation_accuracy':0.0,
    'overall_test_accuracy':0.0,
    
    
    # input shape
    'input_height'  : 224,
    'input_width'  : 224,
    'input_channels' : 3,
    'batch_size'  : 32,
    'dataformat': 'channels_last',
    
    
    #top layers settings
    'create_top_layers' : False,
    'top_layers_dims' : [],
    'activation' : 'relu',
    'drop_rate' : 0.0,
    
    # hyper_settings
    # augmentation
    'with_augmentation': True,
    'shift_range':0.1,
    'rotation_range': 2,


    'pre_trained_weights': None,
    'pooling':None,
    'optimizer': None,
    'device':'gpu_single',
    'gpus_number': 4,  
    
    #training/validation/testing data 
    'classes' : 3,
    'train_classes_names': ['nuts', 'screws', 'washers'],
    'train_data_per_class':[],
    
    'valid_classes_names': [],
    'valid_data_per_class':[],
    
    'test_classes_names':[],
    'test_data_per_class':[],
    
    'accuracy_on_valid_data_per_class':[],
    'accuracy_on_test_data_per_class':[],
    
    
     # file pathes
    'model_save_path' : model_save_path,
    'train_data_path' : train_data_path,
    'valid_data_path' : valid_data_path,
    'test_data_path'  : test_data_path,
    'result_sheet':result_sheet,
    'json_file_path':parameters_json_file_path

}




def create_weights_file_name(parameters, dataformat):
    shape = None
    # handling model name and dataset name
    weights_name = parameters['model_name']+ "_weights_on_"+parameters['Dataset_name']+"_"
    # handling input shape
    if dataformat == 'channels_last':
        shape = [parameters['batch_size'], parameters['input_height'], parameters['input_width'], parameters['input_channels']] # if dataformate() == tf [batch_size, input_height, input_width, input_channels] else [batch_size, input_channels, input_height, input_width]
    else:
        shape = [parameters['batch_size'], parameters['input_channels'], parameters['input_height'], parameters['input_width']] # if dataformate() == tf [batch_size, input_height, input_width, input_channels] else [batch_size, input_channels, input_height, input_width]
    str_shape = "["
    for i in range(len(shape)-1):
        str_shape = str_shape + str(shape[i])+","
    str_shape = str_shape + str(shape[len(shape)-1])+"]"

    weights_name = weights_name +"input_shape_"+ str_shape+ "_"

    # handling top layers
    if(parameters['create_top_layers']):
        weights_name = weights_name + "with_top_layers_"

        str_dims = "["
        for i in range(len(parameters['top_layers_dims'])-1):
            str_dims = str_dims + str(parameters['top_layers_dims'][i])+","
            
        str_dims = str_dims + str(parameters['top_layers_dims'][len(parameters['top_layers_dims'])-1])+"]"
        weights_name = weights_name + str_dims +"_activation_"+parameters['activation']+"_with_rate_"+str(parameters['drop_rate'])+"_"
    else:
        weights_name = weights_name + "NO_top_layers_"

    # handling augmentation and hyper parameters
    # if(parameters['with_augmentation']):
    #     weights_name = weights_name + "with_augmentation_"
    # else:
    #     weights_name = weights_name + "NO_augmentation_"
    
    weights_name = weights_name +"with_pretrained_weights_"+ str(parameters['pre_trained_weights']) +"_pooling_" + str(parameters['pooling']) + ".h5"
    
    return weights_name





def get_parameters(dataformat,_parameters=parameters, model_name=None, Dataset_name =None,
                    input_height=None, input_width=None,input_channels=None ,batch_size=None, total_epochs=None,
                    with_augmentation=True, pre_trained_weights = None, pooling='Max',
                    optimizer=None, create_top_layers=None, top_layers_dims=None, activation=None, rate=None, 
                    _model_save_path=model_save_path, _train_data_path=train_data_path, _valid_data_path=valid_data_path,
                    _result_sheet=result_sheet, device = None, gpus_number = None):

    newParameters = _parameters

    # ------------- Check if the model_name is added or not --------------------
    if(model_name!=None):
        newParameters['model_name'] = model_name
    if(Dataset_name!=None):
        newParameters['Dataset_name'] = Dataset_name
    
    if newParameters['model_name'] == None or  newParameters['Dataset_name'] == None:
        print("Model and dataset name must be added")
        sys.exit(1)
    # ----------------------------------------------------------------------------
    if(device!=None):#default is single GPU
        newParameters['device']=device
    
    if(gpus_number!=None):
        newParameters['gpus_number']=gpus_number

    # fill shape parameters
    if(input_height!=None):
        newParameters['input_height'] = input_height

    if(input_width!=None):
        newParameters['input_width'] = input_width

    if(input_channels!=None):
        newParameters['input_channels']= input_channels

    if(batch_size!=None):
        newParameters['batch_size'] = batch_size

    if(total_epochs!=None):
        newParameters['total_epochs'] = total_epochs

    # fill train data
    if(with_augmentation!=None):
        newParameters['with_augmentation'] = with_augmentation
    
    if(pre_trained_weights!=None):
        newParameters['pre_trained_weights'] = pre_trained_weights
        
    #fill top layer paramters
    if(create_top_layers!=None):
        newParameters['create_top_layers'] = create_top_layers
    
    if(top_layers_dims!=None):
        newParameters['top_layers_dims'] = top_layers_dims
    
    if(activation!=None):
        newParameters['activation'] = activation

    if(rate!=None): # drop out regularization rate
        newParameters['drop_rate'] = rate


    # ------------- Check if the optimizer is added or not --------------------
    if optimizer == None:
        print("Optimizer must be added")
        sys.exit(1)
    else:
        newParameters['optimizer'] = optimizer #
    # --------------------------------------------------------------------------

    newParameters['train_data_path'] = _train_data_path
    newParameters['valid_data_path'] = _valid_data_path
    newParameters['result_sheet'] = _result_sheet
    newParameters['model_save_path'] = _model_save_path + create_weights_file_name(newParameters, dataformat)

    # fill convolution pooling
    newParameters['pooling'] = pooling

    return newParameters




def update_accuracy_in_paramters_on_end_then_save_to_json(parameters, history):
    
    # append function name to the call sequence
    calling_sequence.append("[update_paramters]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function[update_paramters]  in core.py")
    # save parameters in result sheet 

    # get max validation accuracy and the train accuracy according to it.
    max_validation_acc = np.max(history.history['val_acc'])
    index = history.history['val_acc'].index(np.max(history.history['val_acc']))
    train_acc_according_to_max_validation_acc = history.history['acc'][index]

    parameters['train_acc_according_to_max_validation_acc'] = round(train_acc_according_to_max_validation_acc, 4)
    parameters['max_validation_acc'] = round(max_validation_acc,4)
    parameters['best_accuracy_epoch'] = index+1


    parameters['overall_train_accuracy'] = round(history.history['acc'][-1],4)
    parameters['overall_validation_accuracy'] = round(history.history['val_acc'][-1], 4)
    
#     print(parameters)
    write_json_parameters(parameters)

    print(" [INFO] Leaving function[update_paramters]  in core.py")
    print(" ==============================================")




def get_result_parameters(parameters):
    # append function name to the call sequence
    calling_sequence.append("[get_result_parameters]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function [get_result_parameters] in core.py")
    print(" [INFO] form the result csv to write")

    # creat the result dictionary that will be written in a pandas data frame and will be saved as .csv file
    result_sheet = {}
    result_sheet['model_name']= parameters['model_name'] # save model name in case there is more than model "grid search / enforcement learning"
    result_sheet['Dataset_name']= parameters['Dataset_name'] # save data set name in case working with multiversion of a data set

    # save train accuracy where train accuracy will be the train accuracy that match the best validation accuracy // beside the last epoch train accuracy 
    result_sheet['train_accuracy'] = "[train_acc_according_to_max_validation_acc="+str(parameters['train_acc_according_to_max_validation_acc']) + " // overall_train_accuracy=" + str(parameters['overall_train_accuracy'])+"]" #[train_acc_according_to_max_validation_acc/overall_accuracy]
    
    # same goes with validation accuracy as train accuracy
    result_sheet['validation_accuracy'] = "[max_validation_acc="+ str(parameters['max_validation_acc']) + " // overall_validation_accuracy=" + str(parameters['overall_validation_accuracy']) +"]" #[max_validation_acc / over_all_accuracy]
    # save the best epoch that the model got best accuracy on validation set beside the total number of epochs sperated by '//'
    result_sheet['epochs'] = "[best_accuracy_epoch=" + str(parameters['best_accuracy_epoch']) + "// total_epochs=" + str(parameters['total_epochs']) +"]" #[best_epoch / total epochs]
    
    # save the input shape according to keras model dataformat (channels_last/first)
    if parameters['dataformat'] == 'channels_last':
        #[batch, height, width, channels]
        result_sheet['input_shape'] = "[batch_size="+str(parameters['batch_size'])+", height="+str(parameters['input_height'])+", width="+str(parameters['input_width'])+", channels="+str(parameters['input_channels'])+"]" # if dataformate() == tf [batch_size, input_height, input_width, input_channels] else [batch_size, input_channels, input_height, input_width]
    else:
        #[batch, channels, height, width]
        result_sheet['input_shape'] = "[batch_size="+str(parameters['batch_size'])+", channels="+str(parameters['input_channels']) +", height="+str(parameters['input_height'])+", width="+str(parameters['input_width'])+"]" # if dataformate() == tf [batch_size, input_height, input_width, input_channels] else [batch_size, input_channels, input_height, input_width]
      
    #top layers settings
    if(parameters['create_top_layers']): # initial a pretrained model without its top layers and add your own top layers (same as learning transfare concept)
        result_sheet['top_layers_properties'] =  "[top_layers_dims="+str(parameters['top_layers_dims'])+" , activation="+str(parameters['activation'])+ " , rate="+str(parameters['drop_rate'])+"]" # [create_top_layers, top_layers_dims , activation, rate]
    else:
        result_sheet['top_layers_properties'] = "No Top Layers Additional Added"
    
    # hyper_settings
    #[augmentation_status(true/false), with_pre_trained_weights=('imagenet'/None), pooling('Max'/'Avg'/'None')]
    result_sheet['hyper_parameters'] =  "[with_augmentation="+str(parameters['with_augmentation'])+" , with_pre_trained_weights="+str(parameters['pre_trained_weights'])+" , pooling="+str(parameters['pooling'])+"]" # [with_augmentation, pre_trained_weights, pooling]

    #training/validation/testing data 
    result_sheet['classes'] = parameters['classes'] #number of classes 
    result_sheet['train_classes_names'] = parameters['train_classes_names']
    result_sheet['train_data_per_class'] = parameters['train_data_per_class']#number of train examples for each class
    
    result_sheet['valid_classes_names'] = parameters['valid_classes_names']
    result_sheet['valid_data_per_class'] = parameters['valid_data_per_class']#number of validation examples for each class
    
    result_sheet['test_classes_names'] = parameters['test_classes_names']
    result_sheet['test_data_per_class'] = parameters['test_data_per_class']#number of train examples for each class
    
    result_sheet['accuracy_on_valid_data_per_class'] = parameters['accuracy_on_valid_data_per_class']
    result_sheet['accuracy_on_test_data_per_class'] = parameters['accuracy_on_test_data_per_class']
    
    
     # file pathes
    result_sheet['model_save_path'] = parameters['model_save_path']
    # result_sheet['train_data_path'] = parameters['train_data_path']
    # result_sheet['valid_data_path'] = parameters['valid_data_path']
    # result_sheet['result_sheet'] = parameters['result_sheet']

    
    print(" [INFO] Leaving function [get_result_parameters] in core.py")
    print(" ==============================================")

    return result_sheet


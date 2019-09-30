from config import get_parameters, get_result_parameters, update_accuracy_in_paramters_on_end_then_save_to_json
from utils import calling_sequence, get_classes_names_and_data_count_per_class_from_generator, create_input_tensor_shape, np, load_parameters, load_pretrained_model

import os
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential  
from keras import backend as K
# K.tensorflow_backend.set_image_dim_ordering('tf')
# print(K.image_data_format())
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras import optimizers
import pandas as pd

from test_accuaracy import calculate_accuaracy_data_set



# TODO: change Method because it increases the weaights fils size rapidly
def create_top_layers(model, parameters):
    # append function name to the call sequence
    calling_sequence.append("[create_top_layers]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function[create_top_layers]  in core.py")
    print(" ==============================================")

    newModel = Sequential()
    newModel.add(model)
    newModel.add(Flatten(input_shape=model.layers[len(model.layers)-1].output_shape))
    
    print(" [INFO] creating top layers with activation = "+ parameters['activation'])
    for dim in parameters['top_layers_dims']:
            newModel.add(Dense(dim))
            newModel.add(BatchNormalization())
            if(parameters['activation']=='relu'):
                newModel.add(Activation('relu'))
            else:
                newModel.add(LeakyReLU(alpha=0.5))
            
    newModel.add(Dropout(rate=parameters['drop_rate']))    
    newModel.add(Dense(parameters['classes']))
    newModel.add(Activation('softmax'))

    return newModel


 
def train (model,parameters,train_generator, validation_generator,test_generator, parallel=True):
    # append function name to the call sequence
    calling_sequence.append("[train]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function[train]  in core.py")
    # check if parrallel gpus desired or single gpu/cpu
    if parallel == True:
        from keras.utils import multi_gpu_model
        print(" [INFO] running on MULTI GPUS = " + str(parameters['gpus_number']))
        print(K.tensorflow_backend._get_available_gpus())
        model = multi_gpu_model(model, gpus=parameters['gpus_number'])
    
    else:
        print(" [INFO] running on SINGEL GPU/CPU")
    
    
    model.compile(loss='categorical_crossentropy',
            optimizer=parameters['optimizer'],
            metrics=['accuracy'])
    
    history = None
    history = model.fit_generator(
                train_generator,
                steps_per_epoch=train_generator.samples // parameters['batch_size'],
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // parameters['batch_size'],
                epochs=parameters['total_epochs'],
                callbacks=[ModelCheckpoint(parameters['model_save_path'],save_best_only=True)])
    
    print(" [INFO] Leaving function[train]  in core.py")
    print(" ==============================================")

    return history, parameters




def save_train_result(history, parameters, initiate_new_result_sheet = True):
    
    # append function name to the call sequence
    calling_sequence.append("[save_train_result]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function[save_train_result]  in core.py INTENT TO CREATE NEW RESULT SHEET = ", initiate_new_result_sheet)
    # save parameters in result sheet 

    # get max validation accuracy and the train accuracy according to it.
    max_validation_acc = np.max(history.history['val_acc'])
    index = history.history['val_acc'].index(np.max(history.history['val_acc']))
    train_acc_according_to_max_validation_acc = history.history['acc'][index]

    parameters['train_acc_according_to_max_validation_acc'] = round(train_acc_according_to_max_validation_acc, 4)
    parameters['max_validation_acc'] = round(max_validation_acc,4)
    parameters['best_accuracy_epoch'] = index+1


    parameters['overall_train_accuracy'] = round(history.history['acc'][len(history.history['acc'])-1],4)
    parameters['overall_validation_accuracy'] = round(history.history['val_acc'][len(history.history['val_acc'])-1], 4)
    
    # get the paramters format that will be written in csv file
    result_sheet = get_result_parameters(parameters)
    # convert python dict to pandas data frame
    result_sheet = pd.DataFrame([result_sheet])
    
    # CHECK IF CREATING NEW .CSV FILE DESIRED OR APPEND TO CURENT ONE
    if(initiate_new_result_sheet):  
        result_sheet.to_csv(parameters['result_sheet'], index=False)
    else:
        previous_result_sheet = pd.read_csv(parameters['result_sheet'])
        result_sheet = pd.concat([previous_result_sheet, result_sheet], sort=False, ignore_index=True)
        result_sheet.to_csv(parameters['result_sheet'], index=False)

    print('===============================================')
    print(" [INFO] save result at "+str(parameters['model_save_path']))
    print('===================SAVED=======================')
    print(" [INFO] Leaving function[save_train_result]  in core.py")
    print(" ==============================================")
        



def prepare_train_valid_data(parameters, flip=True):
    # append function name to the call sequence
    calling_sequence.append("[prepare_train_valid_data]==>>")
    print(" ==============================================")
    print(" [INFO] Entering function[prepare_train_valid_data]  in core.py")

    # augment train data if specified 
    if(parameters['with_augmentation']):
        train_ImageDataGenerator = ImageDataGenerator(rescale=1./255.0, rotation_range=parameters['rotation_range'],
                                        width_shift_range=parameters['shift_range'], height_shift_range=parameters['shift_range'],
                                        horizontal_flip=flip, vertical_flip=flip)
    else:
        train_ImageDataGenerator = ImageDataGenerator(rescale=1./255.0)
    #-------------------------------------------------------------------------
    # load data in rgb or grayscale according to number of channel of the input tensor for the model
    _class_mode = 'rgb'
    #-------------------------------------------------------------------------
    if(parameters['input_channels']==1):
        _class_mode = 'grayscale'
    #-------------------------------------------------------------------------
    # load train classe data from directory
    train_generator = train_ImageDataGenerator.flow_from_directory(
        parameters['train_data_path'],
        target_size=(parameters['input_height'], parameters['input_width']),
        batch_size=parameters['batch_size'],
        class_mode='categorical', color_mode =_class_mode) # set as training data
    #-------------------------------------------------------------------------
    # load validation data without augmentation
    valid_ImageDataGenerator = ImageDataGenerator(rescale=1./255.0)
    validation_generator = valid_ImageDataGenerator.flow_from_directory(
        parameters['valid_data_path'], # same directory as training data
        target_size=(parameters['input_height'], parameters['input_width']),
        batch_size=parameters['batch_size'],
        class_mode='categorical',color_mode =_class_mode, shuffle=False) # set as validation data
    #-------------------------------------------------------------------------
    test_ImageDataGenerator = ImageDataGenerator(rescale=1./255.0)
    test_generator = test_ImageDataGenerator.flow_from_directory(
        parameters['test_data_path'], # same directory as training data
        target_size=(parameters['input_height'], parameters['input_width']),
        batch_size=parameters['batch_size'],
        class_mode='categorical',color_mode =_class_mode, shuffle=False) # set as validation data
    #-------------------------------------------------------------------------
    train_class_names, train_data_per_class = get_classes_names_and_data_count_per_class_from_generator(train_generator)
    valid_class_names, valid_data_per_class = get_classes_names_and_data_count_per_class_from_generator(validation_generator)
    test_class_names, test_data_per_class = get_classes_names_and_data_count_per_class_from_generator(test_generator)
    #-------------------------------------------------------------------------

    parameters['classes'] = len(train_class_names)            # assign classes count
    #-------------------------------------------------------------------------
    parameters['train_classes_names'] = train_class_names     # assign classes names
    parameters['train_data_per_class'] = train_data_per_class # assign data size for each class
    #-------------------------------------------------------------------------
    parameters['valid_classes_names'] = valid_class_names     # assign classes names
    parameters['valid_data_per_class'] = valid_data_per_class # assign data size for each class
    #-------------------------------------------------------------------------
    parameters['test_classes_names'] = test_class_names       # assign classes names
    parameters['test_data_per_class'] = test_data_per_class   # assign data size for each class
    #-------------------------------------------------------------------------


    print(" [INFO] Leaving function[prepare_train_valid_data]  in core.py")
    print(" ==============================================")
    return train_generator, validation_generator, test_generator, parameters





def constrcut_model (parameters, model):
    # append function name to the call sequence
    calling_sequence.append("[constrcut_model]==>>")
    print(" ==============================================")
    print(" [INFO] Entering function[constrcut_model]  in core.py")
    # get train generator , validation_generator, test_generator, parameters from  prepare_train_valid_data
    # after loading train, validation and test data (classes names, and data for each class)
    train_generator, validation_generator, test_generator, parameters = prepare_train_valid_data(parameters)
    # creat top layer if desired with configuration specified manually in parameters dictionary before
    if(parameters['create_top_layers']):
        model = create_top_layers(model, parameters)
    
    print(" [INFO] Leaving function[constrcut_model]  in core.py")
    print(" ==============================================")
    return model, train_generator, validation_generator, test_generator, parameters




def initiate_model_then_train_then_save_results_to_csv_then_clear(baseModel, parameters, save_new_result_sheet = True):
    # append function name to the call sequence
    calling_sequence.append("[initiate_model_then_train_then_save_results_to_csv_then_clear]==>>")

    print(" ==============================================")
    print(" [INFO] Entering function[initiate_model_then_train_then_save_results_to_csv_then_clear]")
    #-----------------------------------------------------------------------------------------------------------------
    #construct model and prepare data for training
    model, train_generator, validation_generator, test_generator, parameters = constrcut_model (parameters, baseModel)
    #-----------------------------------------------------------------------------------------------------------------
    # check if train on parallel gpus
    if(parameters['device']=='gpu_parallel'):
        print(" [INFO] target multi gpus...")
        _parallel = True
    else:
        print(" [INFO] target single gpu...")
        _parallel =False
    #-----------------------------------------------------------------------------------------------------------------
    # start training 
    history, parameters  = train(model, parameters, train_generator, validation_generator, test_generator, parallel=_parallel)
    #-----------------------------------------------------------------------------------------------------------------
    # apply testset
    # TODO: change this to work with generators
    max_prediction_time, parameters = calculate_accuaracy_data_set(DataPath=parameters['test_data_path'],parameters=parameters, model = model)
    print("max prediction time = ", max_prediction_time)
    #-----------------------------------------------------------------------------------------------------------------
    # save train result 
    save_train_result(history,parameters, initiate_new_result_sheet = save_new_result_sheet)
    #-----------------------------------------------------------------------------------------------------------------
    update_accuracy_in_paramters_on_end_then_save_to_json(parameters, history)
    # clear seassion
    del model, train_generator, validation_generator, parameters, history
    K.clear_session()

    print(" [INFO] calling sequence -> ", calling_sequence)
    calling_sequence.clear()

    print(" [INFO] Leaving function[initiate_model_then_train_then_save_results_to_csv_then_clear]")
    print(" ==============================================")





def load_model_and_continue_training(model_save_path, json_parameters_path, save_new_result_sheet):
    # append function name to the call sequence
    calling_sequence.append("[load_model_and_continue_training]==>>")
    print(" ==============================================")
    print(" [INFO] Entering function[load_model_and_continue_training]  in core.py")
    # -----------------------------------------------------------------------------------------------------
    # load saved parameters
    parameters = load_parameters(json_parameters_path)
    print("loaded parameters", parameters)
    # load saved model
    model = load_pretrained_model(model_path=model_save_path)
    # -----------------------------------------------------------------------------------------------------
    # get train generator , validation_generator, test_generator, parameters from  prepare_train_valid_data
    # after loading train, validation and test data (classes names, and data for each class)
    train_generator, validation_generator, test_generator, parameters = prepare_train_valid_data(parameters)
    # ------------------------------------------------------------------------------------------------------
    # check if train on parallel gpus
    if(parameters['device']=='gpu_parallel'):
        print(" [INFO] target multi gpus...")
        _parallel = True
    else:
        print(" [INFO] target single gpu...")
        _parallel =False
    #-----------------------------------------------------------------------------------------------------------------
    # start training 
    history, parameters  = train(model, parameters, train_generator, validation_generator, test_generator, parallel=_parallel)
    #-----------------------------------------------------------------------------------------------------------------
    # apply testset
    # TODO: change this to work with generators
    max_prediction_time, parameters = calculate_accuaracy_data_set(DataPath=parameters['test_data_path'],parameters=parameters, model = model)
    print("max prediction time = ", max_prediction_time)
    #-----------------------------------------------------------------------------------------------------------------
    # save train result 
    save_train_result(history,parameters, initiate_new_result_sheet = save_new_result_sheet)
    #-----------------------------------------------------------------------------------------------------------------
    update_accuracy_in_paramters_on_end_then_save_to_json(parameters, history)
    #-----------------------------------------------------------------------------------------------------------------
    # clear seassion
    del model, train_generator, validation_generator, parameters, history
    K.clear_session()

    print(" [INFO] calling sequence -> ", calling_sequence)
    calling_sequence.clear()

    print(" [INFO] Leaving function[load_model_and_continue_training]")
    print(" ==============================================")

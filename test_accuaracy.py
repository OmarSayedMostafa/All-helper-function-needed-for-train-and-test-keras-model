
import os, glob, cv2
from config import parameters
from utils import load_pretrained_model
from utils import time, np

model = None


def prepare_img_for_prediction(img_path, parameters):
    # load the image
    if parameters['input_channels']==1:
        img = cv2.imread(img_path, 0) # read gray scale image image
    else:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # -----------------------------------------------------------------------------------
    img  = img /  255.0  # normalise
    img = cv2.resize(img,(parameters['input_width'],parameters['input_height'])) # resize
    # -----------------------------------------------------------------------------------
    if parameters['dataformat'] == 'channels_first':
        img = np.rollaxis(img, 0,2) # get channel first 

    img_tensor = np.expand_dims(img,axis=0) # attach the batch dim

    return img_tensor


def calculate_accuaracy_data_set(DataPath=None, parameters=None, model = model):
    max_time = 0.0
    classes = os.listdir(DataPath) # get all classes names from directories label
    accuracy_per_class =[]
    print(classes)
    for xclass in classes: # loop over folders/classes
        os.system("mkdir "+xclass+"_test_missed")
        os.system("rm -rf "+xclass+"_test_missed/*")
        true_prediction_counter = 0
        false_prediction_counter = 0
        imgs_file_list = glob.glob(os.path.join(os.path.join(DataPath, xclass),'*')) # load all images in a given directory and process them 
        for img_path in imgs_file_list:
            img_tensor = prepare_img_for_prediction(img_path, parameters)
            # predict
            start_time = time.time()
            prediction = model.predict(img_tensor)
            needed_time = time.time()-start_time

            if(needed_time> max_time):
                max_time = needed_time
            
            label = parameters['train_classes_names'][np.argmax(prediction)]
            # print("ground_truth = "+xclass+" // prediction = "+label)
            if(label == xclass):
                true_prediction_counter +=1
            else:
                os.system("cp "+img_path+" "+xclass+"_test_missed/"+label+"_"+str(false_prediction_counter)+".jpg")
                false_prediction_counter +=1

        accuracy = round(float(true_prediction_counter)/float(true_prediction_counter+false_prediction_counter),4)


        accuracy_per_class.append(accuracy)
        print("accuracy over class["+xclass+"]= "+str(accuracy)+" %")

        print("number of hits = " , true_prediction_counter)
        print("number of miss = " , false_prediction_counter)
        print("====================================================")

    parameters['accuracy_on_test_data_per_class'] = accuracy_per_class

    return max_time, parameters




def predict_generator(test_generator):
    "Calculate accuracy based on Test Generator"
    score = model.evaluate_generator(test_generator)
    # Score[0] : Loss , Score[1] : accuracy, e.g :[0.48987591472313957, 0.9185750630065685]
    return score[1]



# model = load_pretrained_model(model_path="./mobile_mahmoud.h5")



# maxtime = calculate_accuaracy_on_valid_test_sets(DataPath='./valid', parameters=parameters)


# print("max prediction time = ", maxtime)
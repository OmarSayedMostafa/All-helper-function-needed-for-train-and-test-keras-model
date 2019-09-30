# All-helper-function-needed-for-train-and-test-keras-model
this is all helper and utilities function needed to train and test a keras model included data set loading and valid/test set evaluation on each class and over all classes , and included too saving result to .csv file and model paramters as .json file

 - Features:
	 - parameters configurations inside config.py
		 - parameters :: python dictionary contains all parameters may needed while training or testing any model, this dictionary is set once before running through function [get_parameters] inside config.py or can be set from previous saved parameters.
		 - ability to save configuration parameters as json to reload it agin and continue training or testing the model
		 - run multi models with the same configuration , or same model with different configuration many times and all training models weights will be saved under different weights file name contains  model name and configuration data within it so you can distinguish between models weights [ en-sample learning]
	 - save train and test result to .csv file to compare accuracy  between multi models results and store many information about the training set (classes names, data count per class, accuracy on valid/test set per class and over all accuracy for train and valid set and test set)
	 - choose to run on single (cpu/gpu) or multi gpus and gpus number if parallel gpus desired.
	 - load saved model and its json saved parameters and continue training / evaluating-testing model 
	 - evaluate accuracy of the model on each class alone to know which class your model has a weak prediction in it.

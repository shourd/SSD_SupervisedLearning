"""
Executes the entire filtering, training and testing process
"""
from data_filter import filter_data
import ssd_trainer
import ssd_predictor

''' SETTINGS '''
model_name = 'first_model'
data_folder = 'output'
epochs = 5

''' START CODE '''
# remove scenarios with LOS or too many resolutions
filter_data(data_folder)

# load the data and train the network, saves model to disk
ssd_trainer.train_model(data_folder, model_name, epochs)

# load model form disk and test data set to determine accuracy
loaded_model = ssd_predictor.load_model(model_name)
accuracy = ssd_predictor.test_existing_data(loaded_model)

print('-------------------')
print('Accuracy: {}'.format(accuracy))

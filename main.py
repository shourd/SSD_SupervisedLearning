"""
Executes the entire filtering, training and testing process
"""
from dataFilter import filter_data
import ssd_trainer
import ssd_predictor

''' SETTINGS '''
model_name = 'testModel_2'
data_folder = 'data'
epochs = 15

''' START CODE '''
# remove scenarios with LOS or too many resolutions
filter_data()

# load the data and train the network, saves model to disk
ssd_trainer.train_model(data_folder, model_name)

# load model form disk and test data set to determine accuracy
loaded_model = ssd_predictor.load_model()
accuracy = ssd_predictor.test_existing_data(loaded_model)

print('-------------------')
print('Accuracy: {}'.format(accuracy))

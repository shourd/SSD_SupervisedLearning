import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ssd_dataloader

new_data = True
show_plots = True

# load json and create model
filename = 'testModel'

json_file = open('{}.json'.format(filename), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('{}.h5'.format(filename))
print("Loaded model from disk")

# load data
num_classes = 2  # number of resolution buckets
size = (120, 120)  # target img dimensions
input_shape = (size[0], size[1], 1)

if new_data:
    x_train = ssd_dataloader.load_SSD(size, 'testData')

    for sample in range(0, len(x_train)):

        image = x_train[None, sample, :, :, :]  # None to retain dimensionality

        prediction = loaded_model.predict(image)
        print('Sample {} Probabilities:'.format(int(sample)))
        print(prediction)
        prediction = int(np.argmax(prediction))

        if prediction == 0:
            print('Left')
        elif prediction == 1:
            print('Right')
        else:
            print('fout')

        if show_plots:
            plt.imshow(image[0, :, :, 0])
            plt.draw()
            plt.show()

else:

    with open('ssdAll.pickle', 'rb') as f:
        x_train = pickle.load(f)

    with open('reso_vector.pickle', 'rb') as f:
        y_train = pickle.load(f)

    # # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    y_train = np.array(y_train)
    y_train = (y_train == "right").astype(int)
    # y_train = keras.utils.to_categorical(y_train, num_classes)  # creates (samples, num_categories) array



    for i in range(0, len(x_train)):
        test_sample = i

        image = x_train[None, test_sample, :, :, :]  # None to retain dimensionality
        target = int(y_train[test_sample])

        prediction = loaded_model.predict(image)
        # print(prediction)
        prediction = int(np.argmax(prediction))

        if target is not prediction:
            print('Fout')
        else:
            print('Goed')
        # print('Prediction:', prediction)
        # print('Actual: ', target)

        if show_plots:
            plt.imshow(image[0, :, :, 0])
            plt.draw()
            plt.show()

# random tester
# test_sample = 10
#
# image = x_test[None, test_sample, :, :, :]  # None to retain dimensionality
# target = int(y_test[test_sample])
#
# prediction = loaded_model.predict(image)
# prediction = int(argmax(prediction))
#
# print('Prediction:', prediction)
# print('Actual: ', target)
#
# plt.imshow(image[0, :, :, 0])
# plt.draw()
# plt.show()







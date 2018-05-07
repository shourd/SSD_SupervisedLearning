import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
from numpy import argmax
import ssd_dataloader

# load json and create model
json_file = open('model_SSD.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_SSD.h5")
print("Loaded model from disk")

# load data
num_classes = 2  # number of resolution buckets
size = (120, 120)  # target img dimensions
input_shape = (size[0], size[1], 1)

x_train = ssd_dataloader.load_SSD(size)
y_train = ssd_dataloader.load_resos(num_classes)
print(y_train.shape)
y_train = argmax(y_train, 1)
print(y_train.shape)

show_plots = True

for i in range(0, 40):
    test_sample = i

    image = x_train[None, test_sample, :, :, :]  # None to retain dimensionality
    target = int(y_train[test_sample])

    prediction = loaded_model.predict(image)
    print(prediction)
    prediction = int(argmax(prediction))

    if target is not prediction:
        print('Fout')
    else:
        print('Goed')
    print('Prediction:', prediction)
    print('Actual: ', target)

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







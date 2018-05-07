import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
from numpy import argmax

# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

# load json and create model
json_file = open('model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_x, img_y = 28, 28
num_classes = 10

# reshape input
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

# reshape output
#y_train_cat = keras.utils.to_categorical(y_train, num_classes)
#y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# evaluate loaded model on test data
# loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# score = loaded_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

for i in range(0, 60000):
    test_sample = i

    image = x_test[None, test_sample, :, :, :]  # None to retain dimensionality
    target = int(y_test[test_sample])

    prediction = loaded_model.predict(image)
    prediction = int(argmax(prediction))

    if target is not prediction:
        print('Fout')
        plt.imshow(image[0, :, :, 0])
        plt.draw()
        print('Prediction:', prediction)
        print('Actual: ', target)
        plt.show()
        pass

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







from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pylab as plt
import ssd_dataloader
from os import  environ
import pickle
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # surpresses the warning that CPU isn't used optimally (VX/FMA functionality)

### TRAINING SETTINGS
batch_size = 128  # for backprop type
epochs = 15

### LOAD SSD AND RESO DATA
num_classes = 2
size = (120, 120)  # target img dimensions
input_shape = (size[0], size[1], 1)

x_train = ssd_dataloader.load_SSD(size)
y_train = ssd_dataloader.load_resos(num_classes)

pickle.dump( x_train, open( "SSDs.p", "wb" ) )

### CREATING THE CNN

print_layer_size = True

model = Sequential()

# Set 0
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
if print_layer_size: print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
if print_layer_size: print(model.output_shape)

# Set 0a
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
if print_layer_size: print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
if print_layer_size: print(model.output_shape)

# Set 1
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
if print_layer_size: print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
if print_layer_size: print(model.output_shape)

# Set 2
model.add(Conv2D(64, (5, 5), activation='relu'))
if print_layer_size: print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2)))
if print_layer_size: print(model.output_shape)

# Flattening and FC
model.add(Flatten())
if print_layer_size: print(model.output_shape)

model.add(Dense(256, activation='relu'))
if print_layer_size: print(model.output_shape)

model.add(Dense(num_classes, activation='softmax'))
if print_layer_size: print(model.output_shape)


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_data=(x_test, y_test),
          callbacks=[history])
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
#
# serialize model to JSON
model_json = model.to_json()
with open("model_SSD.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_SSD.h5")
print("Saved model to disk")


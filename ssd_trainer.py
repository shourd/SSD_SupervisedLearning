from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
import numpy as np
import matplotlib.pylab as plt
import ssd_dataloader
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # surpresses the warning that CPU isn't used optimally (VX/FMA functionality)
import pickle




def train_model(data_folder='data', model_name='testModel', epochs=15):

    """ TRAINING SETTINGS """
    batch_size = 128  # for backprop type
    train_ratio = 0.8  # train / test data
    save_model = True
    print_layer_size = True

    ''' LOAD SSD AND RESO DATA '''
    num_classes = 2
    size = (120, 120)  # target img dimensions
    input_shape = (size[0], size[1], 1)

    try:
        x_train, x_test, y_train, y_test = pickle.load(open("training_data.pickle", "rb"))
        print('Data loaded from Pickle')

    except FileNotFoundError:

        print('Start loading data.')
        x_data = ssd_dataloader.load_SSD(size, data_folder)
        y_data = ssd_dataloader.load_resos(data_folder)

        # # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
        y_data = (y_data > 0).astype(int)  # convert positive headings to 1, negative headings to 0.
        y_data = keras.utils.to_categorical(y_data, num_classes)  # creates (samples, num_categories) array

        # Split train and test data
        train_length = int(train_ratio * len(x_data))
        x_train = x_data[0:train_length, :, :, :]
        x_test = x_data[train_length:, :, :, :]

        y_train = y_data[0:train_length, :]
        y_test = y_data[train_length:, :]

        pickle.dump([x_train, x_test, y_train, y_test], open("training_data.pickle", "wb"))
        print('Data saved to disk.')

    ''' CREATING THE CNN '''

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

    # Output model structure to disk
    plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=False)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # sgd = keras.optimizers.SGD(lr=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

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
              validation_data=(x_test, y_test),
              callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # serialize model to JSON
    if save_model:
        model_json = model.to_json()
        with open('{}.json'.format(model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('{}.h5'.format(model_name))
        print("Saved model to disk")

    plt.plot(range(1, epochs+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    return model


if __name__ == "__main__":
    folder_name = 'output'
    model_name = 'first_model'
    epochs = 6
    train_model(folder_name, model_name, epochs)
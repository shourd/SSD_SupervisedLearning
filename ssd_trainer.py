# from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pylab as plt
import ssd_dataloader
from os import environ
# from os import makedirs
# import shutil
import pickle
import time
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # surpresses the warning that CPU isn't used optimally (VX/FMA functionality)


class Settings:
    def __init__(self):
        self.data_folder        = 'data'
        self.model_name         = 'test_model'
        self.epochs             = 15
        self.train_val_ratio    = 0.8
        self.batch_size         = 128
        self.amnt_input_data    = 3000              # the amount of SSDs to use for training
        self.rotation_range     = 10                # the maxium degree of random rotation for data augmentation
        self.size               = 120, 120          # the pixel dimensions of imported SSDs
        self.num_classes        = 2                # amount of resolution classes
        self.save_model         = True              # save trained model to disk


def train_model(settings):

    """ Start loading SSD and Resolution data """

    input_shape = (settings.size[0], settings.size[1], 1)

    try:
        x_data, y_data = pickle.load(open("training_data.pickle", "rb"))
        print('Data loaded from Pickle')

    except FileNotFoundError:

        print('Start loading data.')
        x_data = ssd_dataloader.load_SSD(settings.size, settings.data_folder)
        y_data = ssd_dataloader.load_resos(settings.data_folder)
        pickle.dump([x_data, y_data], open("training_data.pickle", "wb"))
        print('Data saved to disk.')

    # y_data += 25
    # y_data /= 5

    print(np.unique(y_data))

    # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    y_data = (y_data > 0).astype(int)  # convert positive headings to 1, negative headings to 0.
    y_data = keras.utils.to_categorical(y_data, settings.num_classes)  # creates (samples, num_categories) array

    # Split train and test data
    train_length = int(settings.train_val_ratio * len(x_data))
    x_train = x_data[0:train_length, :, :, :]
    x_test = x_data[train_length:, :, :, :]

    y_train = y_data[0:train_length, :]
    y_test = y_data[train_length:, :]

    ''' CREATING THE CNN '''


    data_length = len(x_train)+len(x_test)
    print('Amount of SSDs: {}'.format(data_length))

    model = Sequential()

    # Set 0
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    # Set 0a
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Set 1
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Set 2
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening and FC
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(settings.num_classes, activation='softmax'))

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

    # For debugging purposes. Exports augmented image data
    export_dir = None
    # if export_dir is not None:
    #     try:
    #         makedirs(export_dir)
    #     except FileExistsError:
    #         shutil.rmtree(export_dir)
    #         makedirs(export_dir)

    train_datagen = ImageDataGenerator(
        rotation_range=settings.rotation_range,
        fill_mode='constant',
        cval=1,  # fill with white pixels [0;1]
    )

    train_generator = train_datagen.flow(
            x_train,
            y_train,
            save_to_dir=export_dir,
            save_prefix='aug',
            save_format='png')

    # measure training time
    start_time = time.time()

    # start training
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) / settings.batch_size,
        epochs=settings.epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    test_loss = round(score[0], 3)
    test_accuracy = round(score[1], 3)
    train_time = int(time.time() - start_time)
    print('Train time: {}s'.format(train_time))
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

    # serialize model to JSON
    if settings.save_model:
        model_json = model.to_json()
        with open('{}.json'.format(settings.model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('{}.h5'.format(settings.model_name))
        print("Saved model to disk")

    plt.plot(range(1, settings.epochs+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    return model, test_accuracy


if __name__ == "__main__":
    settings = Settings()
    train_model(settings)

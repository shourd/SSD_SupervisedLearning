# from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pylab as plt
import ssd_dataloader
from os import environ, path, makedirs
import shutil
import pickle
import time
import seaborn as sns
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # surpresses the warning that CPU isn't used optimally (VX/FMA functionality)
sns.set_style("darkgrid")


class Settings:
    def __init__(self):
        self.input_dir          = 'data'
        self.output_dir         = 'output'
        self.model_name         = 'model'
        self.epochs             = 25
        self.train_val_ratio    = 0.8
        self.batch_size         = 128
        self.num_samples        = 'all'             # the amount of SSDs to use for training [Integer or 'all']
        self.rotation_range     = 0                 # the maxium degree of random rotation for data augmentation
        self.size               = 64, 64            # the pixel dimensions of imported SSDs
        self.num_classes        = 6                 # amount of resolution classes (2, 4, 6, or 12)
        self.max_reso           = 30                # [deg] larger resolutions will be clipped.
        self.architecture       = 0                 # The standard architecture
        self.randomize_fraction = 0                 # Randomize fraction of samples to simulate human randomness
        self.save_model         = True              # save trained model to disk
        self.reload_data        = False             # load data from Pickle [False] or from raw SSDs [True]


def train_model(settings):

    """ Start loading SSD and Resolution data """
    input_shape = (settings.size[0], settings.size[1], 1)
    ratio_int = int(settings.train_val_ratio * 100)
    fraction_print = int(settings.randomize_fraction * 100)
    iteration_name = '{}c_{}s_{}px_{}deg_{}r_{}a_{}f'.format(
        settings.num_classes,
        settings.num_samples,
        settings.size[0],
        settings.rotation_range,
        ratio_int,
        settings.architecture,
        fraction_print)

    filename = 'training_data_{}px.pickle'.format(settings.size[0])
    try:
        x_data, y_data = pickle.load(open(filename, "rb"))
        print('Data loaded from Pickle')
    except FileNotFoundError:
        print('Start loading data.')
        x_data = ssd_dataloader.load_ssd(settings.size, settings.input_dir)
        print('SSDs loaded. Start loading resolutions.')
        y_data = ssd_dataloader.load_resos(settings.input_dir)
        pickle.dump([x_data, y_data], open(filename, "wb"))
        print('Data saved to disk.')

    # plt.hist(y_data,80)
    # plt.xlabel('Resolution HDG')
    # plt.ylabel('Count')
    # plt.show()
    # print(np.unique(y_data))

    # Downsample resolution increments to fit in output classes
    y_data = np.clip(y_data, -settings.max_reso, settings.max_reso)
    reso_resolution = 2 * settings.max_reso / settings.num_classes
    y_data = np.round((y_data + settings.max_reso) / reso_resolution, 0)
    print('Possible resolutions:')
    print(np.unique(reso_resolution * y_data - settings.max_reso))

    # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    # y_data = (y_data > 0).astype(int)  # convert positive headings to 1, negative headings to 0.
    y_data = keras.utils.to_categorical(y_data, settings.num_classes + 1)  # creates (samples, num_categories) array

    # Split train and test data
    train_length = int(settings.train_val_ratio * len(x_data))
    x_train = x_data[0:train_length, :, :, :]
    x_test = x_data[train_length:, :, :, :]

    y_train = y_data[0:train_length, :]
    y_test = y_data[train_length:, :]

    # Define length of data set to be used
    if settings.num_samples is not 'all':
        x_train = x_train[0:settings.num_samples, :, :, :]
        y_train = y_train[0:settings.num_samples, :]

    """" Make certain amount of resolutions random """
    y_train = randomize_resolutions(y_train, settings.randomize_fraction)
    y_test = randomize_resolutions(y_test, settings.randomize_fraction)

    ''' CREATING THE CNN '''

    data_length = len(x_train)+len(x_test)
    print('Amount of SSDs: {} ({} train / {} val)'.format(data_length, len(x_train), len(x_test)))

    model = Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))

    """" ARCHITECTURES WITH TYPES OF LAYERS """
    # BASELINE ARCHITECTURE
    if settings.architecture == 0:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # smaller pooling layer
    if settings.architecture == 1:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # larger 1st conv filter
    if settings.architecture == 2:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # larger 2nd conv filter
    if settings.architecture == 3:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    """" ARCHITECTURES WITH DIFFERENT AMOUNT OF LAYERS """
    # only one layer
    if settings.architecture == 4:
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

    # one layer combo added to archi_1
    if settings.architecture == 5:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # two layer combos acdded to archi_1
    if settings.architecture == 6:
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening and FC
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(settings.num_classes + 1, activation='softmax'))
    model.summary()

    # Output model structure to disk
    if not path.exists(settings.output_dir):
        makedirs(settings.output_dir)

    plot_model(model, to_file='{}/model_structure_{}px_{}a.png'.format(
        settings.output_dir, 
        settings.size[0], 
        settings.architecture),
        show_shapes=True, show_layer_names=False)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # sgd = keras.optimizers.SGD(lr=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    """ CALLBACKS """
    # HISTORY
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    # CHECKPOINTS
    filepath = settings.output_dir + '/' + iteration_name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint, history]

    # For debugging purposes. Exports augmented image data
    export_dir = None
    if export_dir is not None:
        try:
            makedirs(export_dir)
        except FileExistsError:
            shutil.rmtree(export_dir)
            makedirs(export_dir)

    train_datagen = ImageDataGenerator(
        rotation_range=settings.rotation_range,
        fill_mode='constant',
        cval=1,  # fill with white pixels [0;1]
    )

    train_generator = train_datagen.flow(
            x_train,
            y_train,
            batch_size=settings.batch_size,
            save_to_dir=export_dir,
            save_prefix='aug',
            save_format='png')

    # measure training time
    start_time = time.time()

    # start training
    model.fit_generator(
        train_generator,
        steps_per_epoch=5800 / settings.batch_size,  # len(x_train) / settings.batch_size,
        epochs=settings.epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list)

    score = model.evaluate(x_test, y_test, verbose=0)
    test_loss = round(score[0], 3)
    test_accuracy = round(score[1], 3)
    train_time = int(time.time() - start_time)
    print('Train time: {}s'.format(train_time))
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

    """ SAVING MODEL AND RESULTS """

    # Save resolutions to txt file
    filename = settings.output_dir + '/train_acc'
    with open(filename + '.csv', 'a') as f:
        f.write(",".join(map(str, history.acc)))
        f.write("\n")

    # serialize model to JSON
    settings.model_name = settings.output_dir + '/' + iteration_name
    if settings.save_model:
        model_json = model.to_json()
        with open('{}.json'.format(settings.model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights('{}.h5'.format(settings.model_name))
        print("Saved model to disk")

    plt.plot(range(1, settings.epochs+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy of the SSD CNN')
    plt.savefig('{}/{}.png'.format(settings.output_dir, iteration_name), bbox_inches='tight')

    return model, test_accuracy, train_time


def save_train_data(val_acc_list, train_time_list):
    filename = settings.output_dir + '/train_time_list.csv'
    with open(filename, 'a') as f:
        f.write(",".join(map(str, train_time_list)))
        f.write("\n")

    filename = settings.output_dir + '/val_acc_list.csv'
    with open(filename, 'a') as f:
        f.write(",".join(map(str, val_acc_list)))
        f.write("\n")


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def randomize_resolutions(reso_data, fraction_random):
    cap = int(fraction_random * len(reso_data))
    if fraction_random > 0:
        for idx, resolution in enumerate(reso_data):
            if idx < cap:
                reso_data[idx] = np.zeros(settings.num_classes + 1)  # Set all resos to 0
                reso_data[idx][np.random.randint(settings.num_classes+1)] = 1 # set a random reso to 1

    print('{} samples ({}%) randomized!'.format(cap, fraction_random * 100))
    return reso_data


if __name__ == "__main__":
    settings = Settings()
    train_time_list = []
    val_acc_list = []
    settings.input_dir = 'dataset22May'
    settings.output_dir = 'outputRandomness'
    settings.num_samples = 3000

    print('Input dimensions')
    size_list = [(120, 120), (96, 96), (64, 64), (32, 32), (16, 16)]
    for size in size_list:
        print('<--------- DIMENSION: {} px --------->'.format(size[0]))
        settings.size = size
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)
    train_time_list = []
    val_acc_list = []
    settings.size = (96, 96)

    total_architectures = 7
    print('Architectures')
    for architecture_num in range(total_architectures):
        print('<--------- ARCHITECTURE: {} --------->'.format(architecture_num))
        settings.architecture = architecture_num
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)
    train_time_list = []
    val_acc_list = []

    settings.architecture = 0

    print('Rotations')
    rotation_list = [0, 1, 2, 3, 4, 5]
    settings.num_samples = 1000
    for rotation in rotation_list:
        print('<--------- ROTATION: {} deg --------->'.format(rotation))
        settings.rotation_range = rotation
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)
    train_time_list = []
    val_acc_list = []
    settings.num_samples = 'all'
    settings.rotation_range = 0

    print('Num of output classes')
    class_list = [2, 4, 6, 8, 10, 12]
    for num_classes in class_list:
        print('<--------- NUMBER OF CLASSES: {} --------->'.format(num_classes))
        settings.num_classes = num_classes
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)
    train_time_list = []
    val_acc_list = []

    settings.num_classes = 6

    print('Num of samples')
    sample_list = [150, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000, 'all']
    for num_samples in sample_list:
        print('<--------- ITERATION: {} samples --------->'.format(num_samples))
        settings.num_samples = num_samples
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)
    settings.num_samples = 3000

    print('Human Randomizer')
    fraction_list = [0]
    for fraction in fraction_list:
        print('<--------- FRACTION: {} --------->'.format(fraction))
        settings.randomize_fraction = fraction
        _, val_acc, train_time = train_model(settings)
        val_acc_list.append(val_acc)
        train_time_list.append(train_time)

    save_train_data(val_acc_list, train_time_list)

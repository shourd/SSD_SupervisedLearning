""" This file evaluates a neural network model using the test data set """
import keras
from keras.models import model_from_json
from ssd_trainer import Settings
import ssd_dataloader
import pickle
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style("darkgrid")


def load_model(model_name):
    """ Load JSON model from disk """
    print(model_name)
    print("Start loading model.")
    try:
        json_file = open('{}.json'.format(model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('{}.hdf5'.format(model_name))
        print("Loaded model from disk.")
        return model
    except FileNotFoundError:
        print('Model not found.')
        return


def load_data():
    """ Start loading SSD and Resolution data """

    filename = 'test_data_{}px.pickle'.format(settings.size[0])
    if settings.reload_data:
        print('Start loading data.')
        x_test = ssd_dataloader.load_ssd(settings.size, settings.test_dir)
        print('SSDs loaded. Start loading resolutions.')
        y_test = ssd_dataloader.load_resos(settings.test_dir)
        pickle.dump([x_test, y_test], open(filename, "wb"))
        print('Data saved to disk.')
    else:
        try:
            x_test, y_test = pickle.load(open(filename, "rb"))
            print('Data loaded from Pickle')
        except FileNotFoundError:
            print('Start loading data.')
            x_test = ssd_dataloader.load_ssd(settings.size, settings.test_dir)
            print('SSDs loaded. Start loading resolutions.')
            y_test = ssd_dataloader.load_resos(settings.test_dir)
            pickle.dump([x_test, y_test], open(filename, "wb"))
            print('Data saved to disk.')

    y_test = np.clip(y_test, -settings.max_reso, settings.max_reso)
    reso_resolution = 2 * settings.max_reso / settings.num_classes
    y_test = np.round((y_test + settings.max_reso) / reso_resolution, 0)
    y_test = keras.utils.to_categorical(y_test, settings.num_classes + 1)  # creates (samples, num_categories) array

    # settings.num_samples = 500
    # x_test = x_test[0:settings.num_samples, :, :, :]
    # y_test = y_test[0:settings.num_samples, :]

    return x_test, y_test


def evaluate_model(model, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=1, batch_size=128)
    test_accuracy = round(score[1], 3)

    return test_accuracy


def get_model_path():
    settings.ratio_int = int(settings.train_val_ratio * 100)
    model_name = '{}c_{}s_{}px_{}deg_{}r_{}a_{}f'.format(
        settings.num_classes,
        settings.num_samples,
        settings.size[0],
        settings.rotation_range,
        settings.ratio_int,
        settings.architecture,
        settings.randomize_fraction)

    model_path = settings.input_dir + '/' + model_name
    return model_path


def print_and_show(name, labels, acc_list):
    # Save resolutions to txt file
    filename = settings.input_dir + '/test_acc_{}'.format(name)
    with open(filename + '.csv', 'w') as f:
        f.write(",".join(map(str, acc_list)))

    # plot
    figsize = (5, 4)
    plt.figure(figsize=figsize)
    plt.scatter(labels, acc_list)
    # x-axis
    if plot_name == 'classes':
        xlabel = 'Classes'
    elif plot_name == 'dimensions':
        xlabel = 'Input dimensions (px x px)'
    elif plot_name == 'rotations':
        xlabel = 'Maximum rotation during augmentation'
    elif plot_name == 'architectures':
        xlabel = 'Architecture number'
    elif plot_name == 'samples':
        xlabel = 'Input samples'
    elif plot_name == 'randomness':
        xlabel = 'Percent of samples randomized'

    plt.xlabel(xlabel)
    plt.ylabel('Test accuracy')
    plt.title('')
    plt.ylim(0, 1)
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    settings = Settings()
    train_time_list = []
    val_acc_list = []
    settings.num_samples = 'all'
    settings.input_dir = 'outputRandomness'
    settings.test_dir = 'test_data_filtered'

    # acc_list = []
    # print('Input dimensions')
    # sizes = [120, 96, 64, 32, 16]
    # size_list = [(120, 120), (96, 96), (64, 64), (32, 32), (16, 16)]
    # for size in size_list:
    #     print('<--------- DIMENSION: {} px --------->'.format(size[0]))
    #     plot_name = 'dimensions'
    #     settings.size = size
    #     x_test, y_test = load_data()
    #     model_path = get_model_path()
    #     loaded_model = load_model(model_path)
    #     accuracy = evaluate_model(loaded_model, x_test, y_test)
    #     acc_list.append(accuracy)
    #     print('Label:', size)
    #     print('Test accuracy:', accuracy)
    # print_and_show(plot_name, sizes, acc_list)
    #
    # settings.size = (96, 96)
    # x_test, y_test = load_data()
    #
    # acc_list = []
    # total_architectures = 7
    # print('Architectures')
    # for architecture_num in range(total_architectures):
    #     print('<--------- ARCHITECTURE: {} --------->'.format(architecture_num))
    #     plot_name = 'architectures'
    #     settings.architecture = architecture_num
    #     model_path = get_model_path()
    #     loaded_model = load_model(model_path)
    #     accuracy = evaluate_model(loaded_model, x_test, y_test)
    #     acc_list.append(accuracy)
    #     print('Label:', architecture_num)
    #     print('Test accuracy:', accuracy)
    # print_and_show(plot_name, [1, 2, 3, 4, 5, 6, 7], acc_list)
    #
    # settings.architecture = 0
    # acc_list = []
    # print('Rotations')
    # rotation_list = [0, 1, 2, 3, 4, 5]
    # settings.num_samples = 1000
    # for rotation in rotation_list:
    #     print('<--------- ROTATION: {} deg --------->'.format(rotation))
    #     plot_name = 'rotations'
    #     settings.rotation_range = rotation
    #     model_path = get_model_path()
    #     loaded_model = load_model(model_path)
    #     accuracy = evaluate_model(loaded_model, x_test, y_test)
    #     acc_list.append(accuracy)
    #     print('Label:', rotation)
    #     print('Test accuracy:', accuracy)
    # print_and_show(plot_name, rotation_list, acc_list)
    #
    # settings.num_samples = 'all'
    # settings.rotation_range = 0
    #
    # acc_list = []
    # print('Num of output classes')
    # class_list = [2, 4, 6, 8, 10, 12]
    # for num_classes in class_list:
    #     print('<--------- NUMBER OF CLASSES: {} --------->'.format(num_classes))
    #     plot_name = 'classes'
    #     settings.num_classes = num_classes
    #     x_test, y_test = load_data()
    #     model_path = get_model_path()
    #     loaded_model = load_model(model_path)
    #     accuracy = evaluate_model(loaded_model, x_test, y_test)
    #     acc_list.append(accuracy)
    #     print('Label:', num_classes)
    #     print('Test accuracy:', accuracy)
    # print_and_show(plot_name, class_list, acc_list)
    #
    # settings.num_classes = 6
    # x_test, y_test = load_data()
    # acc_list = []
    # print('Num of samples')
    # sample_list = [150, 300, 500, 1000, 1500, 2000, 3000, 'all']
    # for num_samples in sample_list:
    #     print('<--------- ITERATION: {} samples --------->'.format(num_samples))
    #     plot_name = 'samples'
    #     settings.num_samples = num_samples
    #     model_path = get_model_path()
    #     loaded_model = load_model(model_path)
    #     accuracy = evaluate_model(loaded_model, x_test, y_test)
    #     acc_list.append(accuracy)
    #     print('Label:', num_samples)
    #     print('Test accuracy:', accuracy)
    # sample_list[7] = 5900
    # print_and_show(plot_name, sample_list, acc_list)
    #
    settings.num_samples = 3000
    settings.size = (64, 64)

    acc_list = []
    print('Randomness')
    fractions = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_test, y_test = load_data()
    for fraction in fractions:
        print('<--------- FRACTION: {} --------->'.format(fraction))
        plot_name = 'randomness'
        settings.randomize_fraction = fraction
        model_path = get_model_path()
        loaded_model = load_model(model_path)
        accuracy = evaluate_model(loaded_model, x_test, y_test)
        acc_list.append(accuracy)
        print('Label:', fraction)
        print('Test accuracy:', accuracy)
    print_and_show(plot_name, fractions, acc_list)
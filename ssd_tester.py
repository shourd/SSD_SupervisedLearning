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
    print("Start loading model.")
    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('{}.hdf5'.format(model_name))
    print("Loaded model from disk.")

    return model


def load_data():
    """ Start loading SSD and Resolution data """

    filename = 'test_data_{}px.pickle'.format(settings.size[0])
    if settings.reload_data:
        print('Start loading data.')
        x_test = ssd_dataloader.load_ssd(settings.size, settings.input_dir)
        print('SSDs loaded. Start loading resolutions.')
        y_test = ssd_dataloader.load_resos(settings.input_dir)
        pickle.dump([x_test, y_test], open(filename, "wb"))
        print('Data saved to disk.')
    else:
        try:
            x_test, y_test = pickle.load(open(filename, "rb"))
            print('Data loaded from Pickle')
        except FileNotFoundError:
            print('Start loading data.')
            x_test = ssd_dataloader.load_ssd(settings.size, settings.input_dir)
            print('SSDs loaded. Start loading resolutions.')
            y_test = ssd_dataloader.load_resos(settings.input_dir)
            pickle.dump([x_test, y_test], open(filename, "wb"))
            print('Data saved to disk.')

    y_test = np.clip(y_test, -settings.max_reso, settings.max_reso)
    reso_resolution = 2 * settings.max_reso / settings.num_classes
    y_test = np.round((y_test + settings.max_reso) / reso_resolution, 0)
    y_test = keras.utils.to_categorical(y_test, settings.num_classes + 1)  # creates (samples, num_categories) array

    settings.num_samples = 500
    x_test = x_test[0:settings.num_samples, :, :, :]
    y_test = y_test[0:settings.num_samples, :]

    return x_test, y_test


def evaluate_model(model, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=1, batch_size=128)
    test_accuracy = round(score[1], 3)

    return test_accuracy


if __name__ == "__main__":
    settings = Settings()
    settings.size = 60, 60
    settings.input_dir = 'test_data'
    settings.reload_data = True
    settings.num_classes = 4
    settings.model_dir = 'output/val_train_ratio'
    # settings.labels = [100, 300, 500, 1000, 1500, 2000, 3000, 5000]
    settings.labels = [5, 10, 15, 20, 50]

    x_test, y_test = load_data()

    acc_list = []
    for label in settings.labels:
        model_name = '4classes_allsamples_60px_0deg_{}ratio'.format(label)
        model_path = settings.model_dir + '/' + model_name
        loaded_model = load_model(model_path)
        accuracy = evaluate_model(loaded_model, x_test, y_test)
        acc_list.append(accuracy)
        print('Label:', label)
        print('Test accuracy:', accuracy)

    print(acc_list)

    # Save resolutions to txt file
    filename = settings.model_dir + '/test_output'
    with open(filename + '.csv', 'a') as f:
        f.write(",".join(map(str, acc_list)))

    plt.plot(settings.labels, acc_list)
    plt.xlabel('Sample size')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy of the SSD CNNs')
    plt.savefig('output/test_accuracy.png', bbox_inches='tight')
    plt.show()

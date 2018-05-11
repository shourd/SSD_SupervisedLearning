from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ssd_dataloader
import time


''' SETTINGS '''
show_plots = False


def load_model(model_name):
    """ Load JSON model from disk """
    print("Start loading model.")
    t = time.time()
    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('{}.h5'.format(model_name))
    elapsed = time.time() - t
    print("Loaded model from disk. ({} s)".format(elapsed))

    return loaded_model


def test_new_data(model, data_folder='testData'):
    """ Loads new SSDs from a folder and shows the predicted resolution """

    size = (120, 120)  # target img dimensions
    x_train = ssd_dataloader.load_SSD(size, data_folder)

    for sample in range(0, len(x_train)):

        image = x_train[None, sample, :, :, :]  # None to retain dimensionality

        prediction = model.predict(image)
        print('Sample {} Probabilities:'.format(int(sample)))
        print(prediction)
        prediction = int(np.argmax(prediction))

        if prediction == 0:
            print('Left')
        elif prediction == 1:
            print('Right')
        else:
            print('Error')

        if show_plots:
            plt.imshow(image[0, :, :, 0])
            plt.draw()
            plt.show()


def test_existing_data(model):
    """ Data can be obtained by running ssd_dataloader.py """

    with open('ssd_all.pickle', 'rb') as f:
        x_data = pickle.load(f)

    with open('reso_vector.pickle', 'rb') as f:
        y_data = pickle.load(f)

    # # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    y_data = (y_data > 0).astype(int)  # convert positive headings to 1, negative headings to 0.

    error_count = 0
    for i in range(0, len(x_data)):
        test_sample = i

        image = x_data[None, test_sample, :, :, :]  # None to retain dimensionality
        target = int(y_data[test_sample])

        probabilities = model.predict(image)
        print(probabilities)
        prediction = int(np.argmax(probabilities))

        if prediction == 0:
            print('Left')
        else:
            print('Right')

        if target is not prediction:
            print('Fout')
            error_count += 1
        else:
            print('Goed')
        # print('Prediction:', prediction)
        # print('Actual: ', target)

        if show_plots:
            plt.imshow(image[0, :, :, 0])
            plt.draw()
            plt.show()

        accuracy = round((i+1 - error_count) / (i+1), 2)
        print('Accuracy: {}'.format(accuracy))

    return accuracy


if __name__ == "__main__":

    loaded_model = load_model()
    accuracy = test_existing_data(loaded_model)






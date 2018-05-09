from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ssd_dataloader


''' SETTINGS '''
new_data = False  # load new data from folder or take data from Pickles
show_plots = False
model_name = 'testModel'


def load_model():
    """ Load JSON model from disk """
    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('{}.h5'.format(model_name))
    print("Loaded model from disk")

    return loaded_model


def test_new_data(model):

    size = (120, 120)  # target img dimensions
    x_train = ssd_dataloader.load_SSD(size, 'testData')

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
            print('fout')

        if show_plots:
            plt.imshow(image[0, :, :, 0])
            plt.draw()
            plt.show()


def test_existing_data(model):

    with open('ssd_all.pickle', 'rb') as f:
        x_train = pickle.load(f)

    with open('reso_vector.pickle', 'rb') as f:
        y_train = pickle.load(f)

    # # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    y_train = (y_train > 0).astype(int)  # convert positive headings to 1, negative headings to 0.

    error_count = 0
    for i in range(0, len(x_train)):
        test_sample = i

        image = x_train[None, test_sample, :, :, :]  # None to retain dimensionality
        target = int(y_train[test_sample])

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






from glob import glob
from PIL import Image
import keras.utils
import numpy as np


def load_SSD(size):

    filelist = glob('data/*.png')
    ssd_stack = []
    for fname in filelist:
        ssd = Image.open(fname)
        ssd = ssd.convert("L")
        ssd = ssd.resize(size, Image.BILINEAR)
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255

    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], size[0], size[1], 1) # dimensions: (sample_num, x_size, y_size, amount of color bands)

    return ssd_stack


### Load and preprocess Resolution Data
def load_resos(num_classes):
    reso_stack = []
    filelist = glob('data/*.txt')
    for fname in filelist:

        with open(fname, 'r') as file:
            content = file.readlines()

        content.pop(0)  # Remove first line

        for line in content:
            hdg_resolution = float(line.partition(";")[0])  # only take hdg reso

        reso_stack.append(hdg_resolution)

    reso_stack = np.array(reso_stack)

    # # convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
    reso_stack = (reso_stack < -25).astype(int) # -30 becomes 1, -25 becomes 0
    reso_stack = keras.utils.to_categorical(reso_stack, num_classes) # creates (samples, num_categories) array

    return reso_stack
from glob import glob
from PIL import Image
import numpy as np
import pickle
import os


def load_SSD(size, folder='data'):

    # filelist = sorted(filelist, key=os.path.getmtime) # to sort on time modified
    filelist = glob('{}/resolutions*.txt'.format(folder))
    ssd_stack = []
    for fname in filelist:
        fname = fname.replace('resolutions', 'SSD')[:-4]
        fname = glob('{}*'.format(fname))[0]  # take only first entry
        ssd = Image.open(fname)
        ssd = ssd.convert("L") # convert to greyscale
        ssd = ssd.resize(size, Image.BILINEAR)
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255

    # dimensions: (sample_num, x_size, y_size, amount of color bands)
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], size[0], size[1], 1)

    return ssd_stack


''' Load and preprocess Resolution Data '''


def load_resos(folder='data'):
    reso_stack = []
    filelist = glob('{}/resolutions*.txt'.format(folder))
    for fname in filelist:

        with open(fname, 'r') as file:
            content = file.readlines()

        hdg_resolution = float(content[1].partition(";")[0])  # only take first resolution
        reso_stack.append(hdg_resolution)

    reso_stack = np.array(reso_stack)

    return reso_stack


# def file_len(fname):
#     with open(fname) as f:
#         for i, l in enumerate(f):
#             pass
#     return i + 1


if __name__ == "__main__":
    size = 120, 120
    num_classes = 2

    SSD = load_SSD(size)
    reso_vector = load_resos()

    # save data
    pickle.dump(SSD, open("ssd_all.pickle", "wb"))
    pickle.dump(reso_vector, open("reso_vector.pickle", "wb"))

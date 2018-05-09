from glob import glob
import numpy as np
from PIL import Image

def generate_ssd(save_files):
    size = 120, 120
    transpose = {'horizontal': Image.FLIP_LEFT_RIGHT,
                 'vertical':  Image.FLIP_TOP_BOTTOM,
                 'rotate90':  Image.ROTATE_90,
                 'rotate180':  Image.ROTATE_180,
                 'rotate270':  Image.ROTATE_270,
                 'transpose': Image.TRANSPOSE}

    ssd_stack = []
    j = 0
    filelist = sorted(glob('data/*.png'))
    for transpose_name, transpose_method in transpose.items():
        i = 0
        for fname in filelist:
            ssd_img = Image.open(fname)
            ssd_img = ssd_img.convert("L")
            ssd_img = ssd_img.resize(size, Image.BILINEAR)
            ssd_img = ssd_img.transpose(transpose_method)
            new_fname = 'data1/ssd_grey_{}_{}.png'.format(j,i)
            if save_files: ssd_img.save(new_fname)
            ssd_array = np.array(ssd_img)
            ssd_stack.append(ssd_array)
            i += 1
        j += 1
    print('All transposed SSDs saved')

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255
    # print(ssd_stack.shape)
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], size[0], size[1], 1)  # dimensions: (sample_num, x_size, y_size, amount of color bands)
    # print(ssd_stack.shape)

    j = 0
    for transpose_name, transpose_method in transpose.items():
        i = 0
        for fname in filelist:
            ssd_img = Image.open(fname)
            ssd_img = ssd_img.convert("L")
            ssd_img = ssd_img.resize(size, Image.BILINEAR)
            ssd_img = ssd_img.transpose(transpose_method)
            new_fname = 'data1/ssd_grey_{}_{}.png'.format(j,i)
            if save_files: ssd_img.save(new_fname)
            ssd_array = np.array(ssd_img)
            ssd_stack.append(ssd_array)
            i = i + 1
        j = j + 1
    print('All transposed SSDs saved')





generate_ssd(save_files=False)

from keras.preprocessing.image import ImageDataGenerator
import os
import shutil

import_dir = 'testData/'
export_dir = 'augmented'
max_images = 20

shutil.rmtree(export_dir)

shift = 5  # pixels
train_datagen = ImageDataGenerator(
    #rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #vertical_flip=True,
    #zca_whitening=False,
    rotation_range=10,
    #width_shift_range=shift,
    #height_shift_range=shift,
    fill_mode='constant',
    cval=255
)

# test_datagen = ImageDataGenerator(rescale=1./255)

try:
    os.makedirs(export_dir)
except FileExistsError:
    print('folder already exists')
    pass


train_generator = train_datagen.flow_from_directory(
        import_dir,
        target_size=(120, 120),
        batch_size=20,
        class_mode='binary',
        save_to_dir=export_dir,
        save_prefix='aug',
        save_format='png',
        interpolation='bilinear',
        color_mode='grayscale')

max_images -= 2
i = 0
for batch in train_generator:
    i += 1
    if i > max_images:
        break
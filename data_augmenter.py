from keras.preprocessing.image import ImageDataGenerator
import os
import shutil


def augment_data(import_dir='testData', export_dir='augmented'):

    max_images      = 5
    batch_size      = 20
    shift           = 5  # pixels
    size            = 120, 120
    rotation_range  = 10  # degrees

    train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #vertical_flip=True,
        #zca_whitening=False,
        rotation_range=rotation_range,
        #width_shift_range=shift,
        #height_shift_range=shift,
        fill_mode='constant',
        cval=255  # fill with white pixels
    )

    # test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            import_dir,
            target_size=size,
            batch_size=batch_size,
            class_mode='binary',
            save_to_dir=export_dir,
            save_prefix='aug',
            save_format='png',
            interpolation='bilinear',
            color_mode='grayscale')

    # empty current export folder
    try:
        os.makedirs(export_dir)
    except FileExistsError:
        shutil.rmtree(export_dir)
        os.makedirs(export_dir)

    # generated the augmented images
    i = 1
    for batch in train_generator:
        i += 1
        if i > max_images:
            break


if __name__ == "__main__":
    augment_data()
import os

import numpy as np
from keras import backend
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from annotation_loader import load_mapping


class DirectoryIteratorWithTarget(DirectoryIterator):
    def __init__(self, directory, image_data_generator, filename_to_target_mapping: dict, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = None, batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, False, None, None, follow_links)
        self.filename_to_target_mapping = filename_to_target_mapping

    def _get_batches_of_transformed_samples(self, index_array):
        """ Returns the next batch. """

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((self.batch_size,) + self.image_shape, dtype=backend.floatx())
        batch_y = np.zeros((self.batch_size,) + (8,), dtype=backend.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            try:
                fname = self.filenames[j]
                img = image.load_img(os.path.join(self.directory, fname),
                                     target_size=self.target_size)
                x = image.img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                if fname in self.filename_to_target_mapping:
                    batch_x[i] = x
                    batch_y[i] = np.asarray(self.filename_to_target_mapping[fname])
            except:
                print("Error when loading {0}".format(self.filenames[j]))
                pass

        return batch_x, batch_y


if __name__ == "__main__":
    filename_to_target_mapping = load_mapping()
    batch_size = 16
    image_data_generator = DirectoryIteratorWithTarget("data/images", ImageDataGenerator(), filename_to_target_mapping,
                                                       target_size=(192, 108), batch_size=batch_size)

    batch_x, batch_y = image_data_generator.next()

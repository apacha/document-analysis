import argparse
import datetime
from glob import glob
import os

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tqdm import tqdm
import numpy as np

from directory_iterator_with_target import DirectoryIteratorWithTarget, load_mapping
from models.ConfigurationFactory import ConfigurationFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_name", type=str, default="res_net_50_gap",
                        help="The model used for training the network. Run >python models/ConfigurationFactory.py to get a list of all available configurations")
    parser.add_argument("--width", default=400, type=int, help="Width of the input-images for the network in pixel")
    parser.add_argument("--height", default=224, type=int, help="Height of the input-images for the network in pixel")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The minibatch-size for training. Reduce to 8 or 4 if your graphics card runs out of "
                             "memory, but keep high otherwise to speed up training")
    parser.add_argument("--use_relative_coordinates", dest="use_relative_coordinates",
                        action="store_true", help="Specify, if relative coordinates should be used instead of absolute")
    parser.set_defaults(use_relative_coordinates=False)
    parser.add_argument("--standardize", dest="standardize",
                        action="store_true", help="Specify, if the input images should be standardized or not")
    parser.set_defaults(use_relative_coordinates=False)
    flags, unparsed = parser.parse_known_args()

    dataset_directory = flags.dataset_directory
    model_name = flags.model_name
    image_width = flags.width
    image_height = flags.height
    batch_size = flags.batch_size
    use_relative_coordinates = flags.use_relative_coordinates
    standardize = flags.standardize

    filename_to_target_mapping = load_mapping(dataset_directory, use_relative_coordinates)
    start_of_training = datetime.date.today()

    generator = ImageDataGenerator()
    if standardize:
        batch_for_computing_statistics = DirectoryIterator(os.path.join(dataset_directory, "images-training"),
                                                           ImageDataGenerator(), target_size=(224, 400),
                                                           # always use small scale for just computing statistics
                                                           batch_size=1000).next()
        generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        generator.fit(batch_for_computing_statistics[0])
        print("Standardizing with computed mean: {0} and std: {1} for sample of 1000 images from training set"
              .format(generator.mean, generator.std))

    training_data_generator = DirectoryIteratorWithTarget(os.path.join(dataset_directory, "images-training"),
                                                          generator,
                                                          filename_to_target_mapping,
                                                          target_size=(image_height, image_width),
                                                          batch_size=batch_size)
    validation_data_generator = DirectoryIteratorWithTarget(os.path.join(dataset_directory, "images-validation"),
                                                            generator,
                                                            filename_to_target_mapping,
                                                            target_size=(image_height, image_width),
                                                            batch_size=batch_size)
    test_data_generator = DirectoryIteratorWithTarget(os.path.join(dataset_directory, "images-test"),
                                                      generator,
                                                      filename_to_target_mapping,
                                                      target_size=(image_height, image_width),
                                                      batch_size=batch_size)

    training_configuration = ConfigurationFactory.get_configuration_by_name(model_name, image_width, image_height)
    print("Model {0} loaded.".format(training_configuration.name()))
    print(training_configuration.summary())

    training_steps_per_epoch = np.math.ceil(training_data_generator.samples / training_data_generator.batch_size)
    validation_steps_per_epoch = np.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)
    test_steps_per_epoch = np.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)

    model_description = "{0}_{1}_{2}x{3}_{4}{5}".format(start_of_training, training_configuration.name(), image_width,
                                                         image_height,
                                                         "relative" if use_relative_coordinates else "absolute",
                                                         "_standardize" if standardize else "")
    best_model_path = model_description + ".h5"
    monitor_variable = 'val_mean_absolute_error'
    model_checkpoint = ModelCheckpoint(best_model_path, monitor=monitor_variable, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor=monitor_variable,
                               patience=training_configuration.number_of_epochs_before_early_stopping,
                               verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor=monitor_variable,
                                                patience=training_configuration.number_of_epochs_before_reducing_learning_rate,
                                                verbose=1,
                                                factor=training_configuration.learning_rate_reduction_factor,
                                                min_lr=training_configuration.minimum_learning_rate)

    log_directory = "./logs/{0}/".format(model_description)
    tensorboard_callback = TensorBoard(
        log_dir=log_directory,
        batch_size=training_configuration.training_minibatch_size)

    callbacks = [model_checkpoint, early_stop, tensorboard_callback, learning_rate_reduction]

    model = training_configuration.classifier()
    model.summary()

    print("Training on dataset...")
    history = model.fit_generator(
        generator=training_data_generator,
        validation_data=validation_data_generator,
        validation_steps=validation_steps_per_epoch,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_configuration.number_of_epochs,
        workers=4,
        callbacks=callbacks
    )

    print("Loading best model from check-point and testing...")
    best_model = keras.models.load_model(best_model_path)
    evaluation = best_model.evaluate_generator(test_data_generator, steps=test_steps_per_epoch)

    for i in range(len(best_model.metrics_names)):
        current_metric = best_model.metrics_names[i]
        print("{0}: {1:.5f}".format(current_metric, evaluation[i]))

import datetime
from glob import glob
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import numpy as np

from directory_iterator_with_target import DirectoryIteratorWithTarget, load_mapping
from models.ConfigurationFactory import ConfigurationFactory

if __name__ == "__main__":
    filename_to_target_mapping = load_mapping()
    start_of_training = datetime.date.today()
    batch_size = 16
    training_data_generator = DirectoryIteratorWithTarget("data/images-training",
                                                          ImageDataGenerator(),
                                                          filename_to_target_mapping,
                                                          target_size=(224, 400),
                                                          batch_size=batch_size)
    validation_data_generator = DirectoryIteratorWithTarget("data/images-validation",
                                                            ImageDataGenerator(),
                                                            filename_to_target_mapping,
                                                            target_size=(224, 400),
                                                            batch_size=batch_size)

    training_configuration = ConfigurationFactory.get_configuration_by_name("res_net_50", 400, 224)
    training_steps_per_epoch = np.math.ceil(training_data_generator.samples / training_data_generator.batch_size)
    validation_steps_per_epoch = np.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)

    best_model_path = "trained_model.h5"
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

    log_directory = "./logs/{0}_{1}/".format(start_of_training, training_configuration.name())
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

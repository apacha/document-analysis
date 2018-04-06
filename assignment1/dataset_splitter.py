import os
import random
import shutil

from tqdm import tqdm


def copy_video_and_annotation(source_directory, destination_directory, background, image_type, number):
    video_source_path = os.path.join(source_directory, background, "{0}{1}.avi".format(image_type, number))
    video_target_path = os.path.join(destination_directory, background, "{0}{1}.avi".format(image_type, number))
    annotation_source_path = os.path.join(source_directory, background, "{0}{1}.gt.xml".format(image_type, number))
    annotation_target_path = os.path.join(destination_directory, background,
                                          "{0}{1}.gt.xml".format(image_type, number))

    os.makedirs(os.path.dirname(video_target_path), exist_ok=True)
    shutil.copy(video_source_path, video_target_path)
    shutil.copy(annotation_source_path, annotation_target_path)


if __name__ == "__main__":
    dataset_directory = "./data/page-detection"
    training_directory = "./data/page-detection-training"
    validation_directory = "./data/page-detection-validation"
    test_directory = "./data/page-detection-test"

    if os.path.exists(training_directory):
        shutil.rmtree(training_directory)
        os.makedirs(training_directory)
    if os.path.exists(validation_directory):
        shutil.rmtree(validation_directory)
        os.makedirs(validation_directory)
    if os.path.exists(test_directory):
        shutil.rmtree(test_directory)
        os.makedirs(test_directory)

    backgrounds = ["background01", "background02", "background03", "background04", "background05"]
    image_types = ["datasheet", "letter", "magazine", "paper", "patent", "tax"]
    numbers = ["001", "002", "003", "004", "005"]

    for background in tqdm(backgrounds):
        for image_type in image_types:
            test_item = random.choice(numbers)
            validation_item = random.choice(list(set(numbers) - {test_item}))
            training_items = list(set(numbers) - {test_item} - {validation_item})

            copy_video_and_annotation(dataset_directory, test_directory, background, image_type, test_item)
            copy_video_and_annotation(dataset_directory, validation_directory, background, image_type, validation_item)
            for training_item in training_items:
                copy_video_and_annotation(dataset_directory, training_directory, background, image_type, training_item)


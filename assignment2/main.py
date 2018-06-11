import argparse
import os

import annotation_loader
import dataset_splitter
import image_to_lines_converter
from ocr_downloader import IAmPrintedDatasetDownloader
from predict import predict
from train import train_model_for_ocr


def remove_samples_with_incorrect_annotations():
    i_am_printed_directory = os.path.join(dataset_directory, "I AM printed")
    try:
        os.remove(os.path.join(i_am_printed_directory, "a04-006.png"))
        os.remove(os.path.join(i_am_printed_directory, "a04-006.xml"))
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_known_args()

    dataset_directory = flags.dataset_directory

    dataset_downloader = IAmPrintedDatasetDownloader(dataset_directory)
    dataset_downloader.download_and_extract_dataset()
    remove_samples_with_incorrect_annotations()
    dataset_splitter.split_dataset_into_training_and_test_sets(dataset_directory)
    maximal_line_height = image_to_lines_converter.split_images_into_text_lines(dataset_directory)
    text_line_image_to_text_mapping = annotation_loader.load_mapping(dataset_directory)
    annotation_loader.remove_lines_without_matching_annotation(dataset_directory, text_line_image_to_text_mapping)

    image_width, image_height = 1900, 64
    # The longest text-line in our dataset consists of 146 characters
    maximum_number_of_characters_in_longest_text_line = 146
    # The alphabet currently has 77 characters, including special characters
    alphabet_length = 77

    model_path = train_model_for_ocr(dataset_directory, "simple", image_width, image_height, alphabet_length,
                                     maximum_number_of_characters_in_longest_text_line, text_line_image_to_text_mapping)

    predict(dataset_directory, model_path, image_height, image_width,
            maximum_number_of_characters_in_longest_text_line)

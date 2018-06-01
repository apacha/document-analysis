import argparse

import editdistance

import annotation_loader
import dataset_splitter
import image_to_lines_converter
import lines_to_window_converter
import ocr_downloader
from annotation_loader import load_alphabet_from_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_known_args()

    dataset_directory = flags.dataset_directory
    ocr_downloader.download_i_am_printed_database(dataset_directory)

    dataset_splitter.split_dataset_into_training_validation_and_test_sets(dataset_directory)

    maximal_line_height = image_to_lines_converter.split_images_into_text_lines(dataset_directory)

    text_line_image_to_text_mapping = annotation_loader.load_mapping(dataset_directory)
    annotation_loader.remove_lines_without_matching_annotation(dataset_directory, text_line_image_to_text_mapping)

    # split text lines with sliding window
    lines_to_window_converter.sliding_window(dataset_directory, maximal_line_height)

    # To compute edit distance of two words
    edit_dist = editdistance.eval("hallo", "hello")

    # train model
    alphabet = load_alphabet_from_samples(text_line_image_to_text_mapping)

    # dictionary correction

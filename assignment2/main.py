import argparse

import dataset_splitter
import image_to_lines_converter
import lines_to_window_converter
import ocr_downloader

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

    # split text lines with sliding window
    lines_to_window_converter.sliding_window(dataset_directory, maximal_line_height)

    # train model

    # dictionary correction

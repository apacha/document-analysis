import os
from glob import glob
import cv2
from tqdm import tqdm

import prepare_image


def convert_all_images_into_text_line_images(dataset_directory: str, output_directory: str):
    os.makedirs(output_directory, exist_ok=True)

    image_files = glob(os.path.join(dataset_directory, "*.png"))
    highest_line_in_entire_dataset = 0

    for image_file in tqdm(image_files, desc="Processing png file"):
        binarized_image, vertical_line_positions, height_of_highest_line = prepare_image.get_image_lines(image_file)
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        one_based_index = 1

        highest_line_in_entire_dataset = max(highest_line_in_entire_dataset, height_of_highest_line)

        for top_edge, bottom_edge in vertical_line_positions:
            output_name = "{0}-line{1}.png".format(file_name, one_based_index)

            # get and save image part
            image = binarized_image[top_edge:bottom_edge, 0:binarized_image.shape[1]]
            cv2.imwrite(os.path.join(output_directory, output_name), image)
            one_based_index += 1

    return highest_line_in_entire_dataset


def split_images_into_text_lines(dataset_directory: str) -> int:
    print("Converting test-images")
    maximal_test_height = convert_all_images_into_text_line_images(os.path.join(dataset_directory, "test"),
                                                                   os.path.join(dataset_directory, "lines-test"))

    print("Converting training-images")
    maximal_training_height = convert_all_images_into_text_line_images(os.path.join(dataset_directory, "training"),
                                                                       os.path.join(dataset_directory,
                                                                                    "lines-training"))

    return max(maximal_test_height, maximal_training_height)


if __name__ == "__main__":
    maximal_line_height = split_images_into_text_lines("data")
    print("Maximal line height: {0}".format(maximal_line_height))

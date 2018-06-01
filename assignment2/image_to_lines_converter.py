import os
from glob import glob
import cv2
from tqdm import tqdm

import prepare_image


def convert_all_images_into_text_line_images(dataset_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    image_files = glob(os.path.join(dataset_directory, "*.png"))
    max_height = 0

    for image_file in tqdm(image_files, desc="Processing png file"):
        [binarized_image, lines, maxImg] = prepare_image.get_image_lines(image_file)
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        one_based_index = 1

        if (maxImg > max_height):
            max_height = maxImg

        for line in range(0, lines.shape[0]):
            output_name = "{0}-line{1}.png".format(file_name, one_based_index)

            # get and save image part
            image = binarized_image[int(lines[line, 0]):int(lines[line, 1]), 0:binarized_image.shape[1]]
            cv2.imwrite(os.path.join(output_directory, output_name), image)
            one_based_index += 1

    return max_height


def load_image_to_line_converter(dataset_directory: str):
    print("Converting test-images")
    maximal_test_height = convert_all_images_into_text_line_images(os.path.join(dataset_directory, "test"),
                                                                   os.path.join(dataset_directory, "lines-test",
                                                                                "all_objects"))

    print("Converting validation-images")
    maximal_validation_height = convert_all_images_into_text_line_images(os.path.join(dataset_directory, "validation"),
                                                                         os.path.join(dataset_directory,
                                                                                      "lines-validation",
                                                                                      "all_objects"))

    print("Converting training-images")
    maximal_training_height = convert_all_images_into_text_line_images(os.path.join(dataset_directory, "training"),
                                                                       os.path.join(dataset_directory, "lines-training",
                                                                                    "all_objects"))

    return max(maximal_test_height, maximal_training_height, maximal_validation_height)


if __name__ == "__main__":
    load_image_to_line_converter("data")

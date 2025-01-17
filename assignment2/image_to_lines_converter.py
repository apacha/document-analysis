import os
from glob import glob
import cv2
from tqdm import tqdm
from typing import Tuple, List

import cv2
import numpy as np


def _load_grayscale_image_and_binarized_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarize image and flip the foreground and background (= white text and black background) by thresholding image
    retval, binarized_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return grayscale_image, binarized_image


def _find_text_lines(binarized_image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """
        Projection Profiles method - histogram profile
        mark the start and end line of each text line and return the image and line indeces
    """
    vertical_histogram_projection = cv2.reduce(binarized_image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    annotated_image = np.copy(binarized_image)

    inside_of_a_textline = False
    line_top_and_bottom_edges = np.zeros((vertical_histogram_projection.size, 1))

    for x in range(0, vertical_histogram_projection.size):
        if vertical_histogram_projection[x] > 0 and not inside_of_a_textline:
            line_top_and_bottom_edges[x] = 1
            inside_of_a_textline = True

        if vertical_histogram_projection[x] == 0 and inside_of_a_textline:
            line_top_and_bottom_edges[x] = -1
            inside_of_a_textline = False

    # Make sure to terminate the last line, in case we are still inside a textline
    if inside_of_a_textline:
        line_top_and_bottom_edges[vertical_histogram_projection.size - 1] = -1

    line_top_edges = np.argwhere(line_top_and_bottom_edges == 1)[:, 0]
    line_bottom_edges = np.argwhere(line_top_and_bottom_edges == -1)[:, 0]
    if len(line_bottom_edges) > len(line_top_edges):
        line_bottom_edges = line_bottom_edges[0:len(line_bottom_edges) - 1]

    line_heights = line_bottom_edges - line_top_edges
    vertical_line_positions = []
    height_of_highest_line = 0
    minimum_line_height = 20
    maximum_line_height = 60

    for x in range(0, line_heights.size):
        if minimum_line_height < line_heights[x] < maximum_line_height:
            annotated_image = cv2.line(annotated_image, (0, line_top_edges[x]),
                                       (annotated_image.shape[1], line_top_edges[x]), (255, 0, 0), 1)
            annotated_image = cv2.line(annotated_image, (0, line_bottom_edges[x]),
                                       (annotated_image.shape[1], line_bottom_edges[x]), (255, 0, 0), 1)
            vertical_line_positions.append((line_top_edges[x], line_bottom_edges[x]))
            height_of_highest_line = max(height_of_highest_line, line_heights[x])

    return annotated_image, vertical_line_positions, height_of_highest_line


def _get_image_lines(image_path: str, visualize=False) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    grayscale_image, binarized_image = _load_grayscale_image_and_binarized_image(image_path)
    image_with_line_annotations, vertical_line_positions, height_of_highest_line = _find_text_lines(binarized_image)
    if visualize:
        cv2.imshow("binarized image", binarized_image)
        cv2.imshow("image_with_lines", image_with_line_annotations)
        cv2.waitKey(0)

    return binarized_image, vertical_line_positions, height_of_highest_line


def _convert_all_images_into_text_line_images(dataset_directory: str, output_directory: str):
    os.makedirs(output_directory, exist_ok=True)

    image_files = glob(os.path.join(dataset_directory, "*.png"))
    highest_line_in_entire_dataset = 0

    for image_file in tqdm(image_files, desc="Processing png file"):
        binarized_image, vertical_line_positions, height_of_highest_line = _get_image_lines(image_file)
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
    maximal_test_height = _convert_all_images_into_text_line_images(os.path.join(dataset_directory, "test"),
                                                                    os.path.join(dataset_directory, "lines-test"))

    print("Converting training-images")
    maximal_training_height = _convert_all_images_into_text_line_images(os.path.join(dataset_directory, "training"),
                                                                        os.path.join(dataset_directory,
                                                                                    "lines-training"))

    return max(maximal_test_height, maximal_training_height)


if __name__ == "__main__":
    maximal_line_height = split_images_into_text_lines("data")
    print("Maximal line height: {0}".format(maximal_line_height))

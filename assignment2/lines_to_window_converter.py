from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import os
from keras.preprocessing import image


def split_line_images_with_sliding_window(input_directory: str, output_directory: str, maximal_line_height):
    os.makedirs(output_directory, exist_ok=True)

    line_images = glob(os.path.join(input_directory, "*.png"))
    sliding_window_width = int(maximal_line_height * 2)

    for line_image in tqdm(line_images, desc="Processing png line-files"):
        grayscale_image = read_grayscale_image_and_add_padding(line_image, maximal_line_height)
        image_width = grayscale_image.shape[1]
        file_name = os.path.splitext(os.path.basename(line_image))[0]
        one_based_index = 1
        i = 0

        while (image_width - int(sliding_window_width / 2) * (i - 1) - sliding_window_width) > 0:
            sliding_window_image = grayscale_image[:, (i - 1) * int(sliding_window_width / 2):(i - 1) * int(
                sliding_window_width / 2) + sliding_window_width]
            i += 1

            # save only positive examples
            # positive examples ... window with text
            # negative example ... window without text
            if np.count_nonzero(sliding_window_image) > 0:
                output_name = "{0}-sw{1}.png".format(file_name, one_based_index)
                cv2.imwrite(os.path.join(output_directory, output_name), sliding_window_image)
                one_based_index += 1


# set same height to each text line
def read_grayscale_image_and_add_padding(path_to_line_image, padded_height) -> np.ndarray:
    img = image.load_img(path_to_line_image, grayscale=True)

    grayscale_image = np.expand_dims(img, 2)

    height, width, channels = grayscale_image.shape
    pad_top = int((padded_height - height) / 2)
    pad_bottom = padded_height - height - pad_top

    pad_top = np.zeros((pad_top, width, 1))
    pad_bottom = np.zeros((pad_bottom, width, 1))

    grayscale_image = np.concatenate((pad_top, grayscale_image, pad_bottom), axis=0)

    return grayscale_image


def split_dataset_with_sliding_window(dataset_directory: str, max: int):
    print("Converting test-images")
    split_line_images_with_sliding_window(os.path.join(dataset_directory, "lines-test"),
                                          os.path.join(dataset_directory, "lines-sliding-window-test"), max)

    print("Converting validation-images")
    split_line_images_with_sliding_window(os.path.join(dataset_directory, "lines-validation"),
                                          os.path.join(dataset_directory, "lines-sliding-window-validation"), max)

    print("Converting training-images")
    split_line_images_with_sliding_window(os.path.join(dataset_directory, "lines-training"),
                                          os.path.join(dataset_directory, "lines-sliding-window-training"), max)


if __name__ == "__main__":
    split_dataset_with_sliding_window("data", 70)

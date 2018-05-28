import os
from glob import glob
import cv2
from tqdm import tqdm

import prepare_image

def convert_all_images_into_textLineImages(dataset_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    image_files = glob(os.path.join(dataset_directory, "*.png"))

    for image_file in tqdm(image_files, desc="Processing png file"):
        lines = prepare_image.get_line_index(image_file)
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        one_based_index = 1

        for line in range (0,lines.shape[0]):
            output_name = "{0}-line{1}.jpg".format(file_name, one_based_index)
            image = get_image_part(image_file)
            cv2.imwrite(os.path.join(output_directory, output_name), image)
            one_based_index += 1
        #
        # success, image = vidcap.read()
        #


def get_image_part(img):
    print("test")



def load_image_to_line_converter(dataset_directory: str):
    print("Converting test-images")
    convert_all_images_into_textLineImages(os.path.join(dataset_directory, "I AM printed-test"),
                                         os.path.join(dataset_directory, "lines-test", "all_objects"))
    print("Converting validation-images")
    convert_all_images_into_textLineImages(os.path.join(dataset_directory, "I AM printed-validation"),
                                         os.path.join(dataset_directory, "lines-validation", "all_objects"))
    print("Converting training-images")
    convert_all_images_into_textLineImages(os.path.join(dataset_directory, "I AM printed-training"),
                                         os.path.join(dataset_directory, "lines-training", "all_objects"))


if __name__ == "__main__":
    load_image_to_line_converter("data")

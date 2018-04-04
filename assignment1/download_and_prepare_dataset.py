import os
from glob import glob
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    dataset_directory = "data"

    backgrounds = os.listdir(os.path.join(dataset_directory, "page-detection"))

    for background in tqdm(backgrounds, desc="Processing backgrounds"):
        avi_files = glob(os.path.join(dataset_directory, "page-detection", background, "*.avi"))
        annotation_files = glob(os.path.join(dataset_directory, "page-detection", background, "*.xml"))

        output_directory = os.path.join(dataset_directory, "images", background)
        os.makedirs(output_directory, exist_ok=True)

        for avi_file in tqdm(avi_files, desc="Processing avi file"):
            vidcap = cv2.VideoCapture(avi_file)
            file_name = os.path.splitext(os.path.basename(avi_file))[0]
            success, image = vidcap.read()
            one_based_index = 1
            success = True

            while success:
                success, image = vidcap.read()
                output_name = "{0}-frame{1}.jpg".format(file_name, one_based_index)
                cv2.imwrite(os.path.join(output_directory, output_name), image)
                one_based_index += 1

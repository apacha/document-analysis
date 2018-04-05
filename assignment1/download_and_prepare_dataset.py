import os
from glob import glob
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    dataset_directory = "data"
    output_directory = os.path.join(dataset_directory, "images", "all_objects")
    os.makedirs(output_directory, exist_ok=True)

    backgrounds = os.listdir(os.path.join(dataset_directory, "page-detection"))

    for background in tqdm(backgrounds, desc="Processing backgrounds"):
        avi_files = glob(os.path.join(dataset_directory, "page-detection", background, "*.avi"))
        annotation_files = glob(os.path.join(dataset_directory, "page-detection", background, "*.xml"))

        for avi_file in tqdm(avi_files, desc="Processing avi file"):
            vidcap = cv2.VideoCapture(avi_file)
            file_name = os.path.splitext(os.path.basename(avi_file))[0]
            success, image = vidcap.read()
            one_based_index = 1
            success = True

            while success:
                output_name = "{0}-{1}-frame{2}.jpg".format(background, file_name, one_based_index)
                cv2.imwrite(os.path.join(output_directory, output_name), image)
                success, image = vidcap.read()
                one_based_index += 1

import os
from glob import glob
import cv2
from tqdm import tqdm


def convert_all_videos_into_frame_images(dataset_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    backgrounds = os.listdir(os.path.join(dataset_directory))
    for background in tqdm(backgrounds, desc="Processing backgrounds"):
        avi_files = glob(os.path.join(dataset_directory, background, "*.avi"))

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


if __name__ == "__main__":
    # convert_all_videos_into_frame_images(os.path.join("data", "page-detection"),
    #                                      os.path.join("data", "images", "all_objects"))
    convert_all_videos_into_frame_images(os.path.join("data", "page-detection-test"),
                                         os.path.join("data", "images-test", "all_objects"))
    convert_all_videos_into_frame_images(os.path.join("data", "page-detection-validation"),
                                         os.path.join("data", "images-validation", "all_objects"))
    convert_all_videos_into_frame_images(os.path.join("data", "page-detection-training"),
                                         os.path.join("data", "images-training", "all_objects"))

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


def load_video_to_image_converter(dataset_directory: str):
    print("Converting test-videos")
    convert_all_videos_into_frame_images(os.path.join(dataset_directory, "page-detection-test"),
                                         os.path.join(dataset_directory, "images-test", "all_objects"))
    print("Converting validation-videos")
    convert_all_videos_into_frame_images(os.path.join(dataset_directory, "page-detection-validation"),
                                         os.path.join(dataset_directory, "images-validation", "all_objects"))
    print("Converting training-videos")
    convert_all_videos_into_frame_images(os.path.join(dataset_directory, "page-detection-training"),
                                         os.path.join(dataset_directory, "images-training", "all_objects"))


if __name__ == "__main__":
    load_video_to_image_converter("data")

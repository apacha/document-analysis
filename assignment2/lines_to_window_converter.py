from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import os

def convert_all_lines_into_window(dataset_directory, output_directory, max):
    os.makedirs(output_directory, exist_ok=True)

    line_files = glob(os.path.join(dataset_directory,"all_objects", "*.png"))
    swWidth = int(max/2)
    swHeight = max


    for image_file in tqdm(line_files, desc="Processing png line-file"):
        img = add_padding(image_file,max)
        imgWidth = img.shape[1]
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        one_based_index = 1
        i=0

        while (imgWidth-int(swWidth/2)*(i-1)-swWidth)>0:
            swImage = img[:,(i-1)*int(swWidth/2):(i-1)*int(swWidth/2)+swWidth]
            i+=1

            # save only positive examples
            # positive examples ... window with text
            # negative example ... window without text
            if np.count_nonzero(swImage)>0:
                output_name = "{0}-sw{1}.png".format(file_name, one_based_index)
                cv2.imwrite(os.path.join(output_directory, output_name), swImage)
                one_based_index +=1

    return

# set same height to each text line
def add_padding(line, max):
    img = cv2.imread(line)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height,width = img.shape
    padT = int((max - height)/2)
    padB = max - height - padT

    padT = np.zeros((padT,width))
    padB = np.zeros((padB,width))

    img = np.concatenate((img, padT), axis=0)
    img = np.concatenate((img, padB), axis=0)

    return img


def load_line_to_window_converter(dataset_directory: str, max: int):
    print("Converting test-images")
    convert_all_lines_into_window(os.path.join(dataset_directory, "lines-test"),
                                         os.path.join(dataset_directory, "lines-sw-test", "all_objects"),max)


    print("Converting validation-images")
    convert_all_lines_into_window(os.path.join(dataset_directory, "lines-validation"),
                                         os.path.join(dataset_directory, "lines-sw-validation", "all_objects"),max)


    print("Converting training-images")
    convert_all_lines_into_window(os.path.join(dataset_directory, "lines-training"),
                                         os.path.join(dataset_directory, "lines-sw-training", "all_objects"),max)



if __name__ == "__main__":
    add_padding("a01-000u-line1.png",58)
    load_line_to_window_converter("data", 58)
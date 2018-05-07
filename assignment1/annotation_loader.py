import os
import xml.etree.ElementTree as ET
from glob import glob

from PIL import Image
from tqdm import tqdm


def load_mapping(dataset_directory="data", use_relative_coordinates=False):
    backgrounds = os.listdir(os.path.join(dataset_directory, "page-detection"))
    filename_to_target_mapping = {}

    rejection_counter = 0
    all_images = glob(os.path.join(dataset_directory, "images-*", "**/*.jpg"))

    for background in tqdm(backgrounds, desc="Loading annotations from background..."):
        annotation_files = glob(os.path.join(dataset_directory, "page-detection", background, "*.gt.xml"))

        for annotation_file in annotation_files:
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            for frame in root.iter("frame"):
                frame_index = frame.attrib["index"]
                rejected = frame.attrib["rejected"]
                filename = "{0}-{1}-frame{2}.jpg".format(background, os.path.splitext(
                    os.path.splitext(os.path.basename(annotation_file))[0])[0], frame_index)
                file_path = [f for f in all_images if filename in f][0]
                width = height = 1.0
                if use_relative_coordinates:
                    img = Image.open(file_path)
                    width, height = img.size
                if rejected != "false":
                    rejection_counter += 1
                points = frame.findall("point")
                target = []
                for point in points:
                    target.append(float(point.attrib["x"]) / width)
                    target.append(float(point.attrib["y"]) / height)
                filename_to_target_mapping[filename] = target

    print("Rejected {0} files".format(rejection_counter))
    return filename_to_target_mapping


if __name__ == "__main__":
    filename_to_target_mapping = load_mapping("data", True)
    print(len(filename_to_target_mapping))

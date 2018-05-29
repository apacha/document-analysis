import os
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm


def load_mapping(dataset_directory="data", use_relative_coordinates=False):
    filename_to_target_mapping = {}

    all_images = glob(os.path.join(dataset_directory, "lines-*", "**/*.png"))

    annotation_files = glob(os.path.join(dataset_directory, "I AM printed", "*.xml"))

    for annotation_file in tqdm(annotation_files, desc="Loading annotations ..."):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        line_index = 1

        for machinePrinted in root.iter("machine-printed-part"):
            lines = machinePrinted.findall("machine-print-line")
            for line in lines:
                filename = "{0}-line{1}.png".format(os.path.splitext(
                    os.path.splitext(os.path.basename(annotation_file))[0])[0], line_index)
                file_path = [f for f in all_images if filename in f][0]

                # if use_relative_coordinates:
                #     img = Image.open(file_path)
                #     width, height = img.size

                filename_to_target_mapping[filename] = line.attrib["text"]
                line_index += 1

    return filename_to_target_mapping


if __name__ == "__main__":
    filename_to_target_mapping = load_mapping("data", True)
    print(len(filename_to_target_mapping))

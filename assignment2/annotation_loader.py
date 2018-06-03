import os
import xml.etree.ElementTree as ET

from glob import glob
from typing import Dict

from tqdm import tqdm


def load_mapping(dataset_directory: str = "data") -> Dict[str, str]:
    text_line_image_to_text_mapping = {}

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

                text_line_image_to_text_mapping[filename] = line.attrib["text"]
                line_index += 1

    return text_line_image_to_text_mapping


def remove_lines_without_matching_annotation(dataset_directory: str, text_line_image_to_text_mapping: Dict[str, str]):
    all_images = glob(os.path.join(dataset_directory, "lines-*/*.png"))
    image_names = [os.path.basename(path) for path in all_images]
    mapped_image_names = list(text_line_image_to_text_mapping.keys())
    unmapped_images = list(set(image_names) - set(mapped_image_names))
    print("Found {0} files without matching annotation".format(len(unmapped_images)))
    for unmapped_image in unmapped_images:
        unmapped_image_path = [path for path in all_images if unmapped_image in path][0]
        print("Removing: " + unmapped_image_path)
        os.remove(unmapped_image_path)


def load_alphabet_from_samples(text_line_image_to_text_mapping: Dict[str, str]) -> str:
    all_strings = list(text_line_image_to_text_mapping.values())
    alphabet = set()
    for string in all_strings:
        characters = list(string)
        for character in characters:
            alphabet.add(character)
    alphabet = list(alphabet)
    alphabet.sort()
    alphabet_string = "".join(alphabet)
    return alphabet_string


if __name__ == "__main__":
    filename_to_target_mapping = load_mapping("data")
    print(len(filename_to_target_mapping))

import argparse
import os
from glob import glob

import keras
from PIL import Image
from PIL.ImageDraw import ImageDraw
from keras.backend import epsilon
from keras.preprocessing import image
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

import annotation_loader


def annotation_to_polygon(annotation_coordinates) -> Polygon:
    """ Converts a list of coordinates for the four corners into a Shapely-Polygon """
    c = annotation_coordinates
    return Polygon([(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])])


def jaccard_index(ground_truth, prediction) -> float:
    ground_truth_polygon = annotation_to_polygon(ground_truth)
    predicted_polygon = annotation_to_polygon(prediction)
    jaccard_index = ground_truth_polygon.intersection(predicted_polygon).area / ground_truth_polygon.union(
        predicted_polygon).area
    return jaccard_index


def predict(dataset_directory: str, model_path: str, image_width: int, image_height: int,use_relative_coordinates: bool, standardize: bool):
    images = glob(dataset_directory + "/images-test/**/*.jpg")

    mean = np.asarray([47.934902, 47.934902, 47.934902])
    std = np.asarray([57.233334, 47.60158, 59.661304])
    mapping = annotation_loader.load_mapping()
    print("Loading model...")
    best_model = keras.models.load_model(model_path)
    output_directory = "detections_output"
    os.makedirs(output_directory, exist_ok=True)
    detections = []
    jaccard_indices = []

    for input_image in tqdm(images, "Detecting pages..."):
        annotation = mapping[os.path.basename(input_image)]
        img = image.load_img(input_image, target_size=(image_height, image_width))
        # We trained with batches, since we only insert one image, we have to add one extra dimension with reshape
        x = np.reshape(image.img_to_array(img), (1, image_height, image_width, 3))
        if standardize:
            x -= mean
            x /= std + epsilon()

        prediction = best_model.predict(x).flatten()

        full_image = Image.open(input_image)
        width, height = full_image.size

        if use_relative_coordinates:
            x_positions = prediction[0::2] * width  # take every second element, starting at position 0
            y_positions = prediction[1::2] * height  # take every second element, starting at position 1
            # stack them into 2d-array for zipping and then flatten array to get the original 8 coordinates
            prediction = np.dstack((x_positions, y_positions)).flatten()

        image_draw = ImageDraw(full_image, 'RGBA')
        image_draw.polygon(annotation, fill=(0, 255, 0, 80))
        image_draw.polygon(prediction, fill=(255, 0, 0, 80))
        try:
            jaccard_indices.append(jaccard_index(annotation, prediction))
        except Exception as ex:
            print("Could not compute Jaccard index for {0} and {1} from image {2}\n{3}".format(annotation, prediction,
                                                                                               input_image, ex))
        detections.append((input_image,) + tuple(list(prediction)))
        full_image.save(
            os.path.join(output_directory, os.path.splitext(os.path.basename(input_image))[0] + "-detect.jpg"))

    pd.DataFrame(detections).to_csv("detections.csv", index=False,
                                    header=["path", "bl_x", "bl_y", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y"])

    average_jaccard_index = sum(jaccard_indices) / float(len(jaccard_indices))
    print("Average Jaccard Index: {0}".format(average_jaccard_index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_path", type=str, default="2018-05-07_res_net_50_gap_400x224_absolute.h5",
                        help="The trained model")
    parser.add_argument("--use_relative_coordinates", dest="use_relative_coordinates",
                        action="store_true", help="Specify, if relative coordinates should be used instead of absolute")
    parser.set_defaults(use_relative_coordinates=False)
    parser.add_argument("--standardize", dest="standardize",
                        action="store_true", help="Specify, if the input images should be standardized or not")
    parser.set_defaults(standardize=False)
    parser.add_argument("--width", default=400, type=int, help="Width of the input-images for the network in pixel")
    parser.add_argument("--height", default=224, type=int, help="Height of the input-images for the network in pixel")

    flags, unparsed = parser.parse_known_args()

    predict(flags.dataset_directory,flags.model_path, flags.width, flags.height,flags.use_relative_coordinates, flags.standardize)



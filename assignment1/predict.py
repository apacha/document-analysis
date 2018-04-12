import os

import keras
from PIL import Image
from PIL.ImageDraw import ImageDraw
from keras.preprocessing import image
import numpy as np

import annotation_loader

if __name__ == "__main__":
    images = ["data/images-test/all_objects/background01-letter005-frame112.jpg",
              "data/images-test/all_objects/background02-magazine004-frame89.jpg",
              "data/images-test/all_objects/background03-paper003-frame53.jpg",
              "data/images-test/all_objects/background04-magazine005-frame88.jpg",
              "data/images-test/all_objects/background05-datasheet005-frame44.jpg"]
    model_path = "2018-04-12_res_net_50_2.h5"
    use_relative_coordinates = True
    mapping = annotation_loader.load_mapping()
    print("Loading model...")
    best_model = keras.models.load_model(model_path)

    for input_image in images:
        annotation = mapping["all_objects\\" + os.path.basename(input_image)]
        img = image.load_img(input_image, target_size=(224, 400))
        # We trained with batches, since we only insert one image, we have to add one extra dimension with reshape
        x = np.reshape(image.img_to_array(img), (1, 224, 400, 3))
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
        full_image.save(os.path.splitext(os.path.basename(input_image))[0] + "-detect.jpg")

import os

import keras
from PIL import Image
from PIL.ImageDraw import ImageDraw
from keras.preprocessing import image
import numpy as np

import annotation_loader

if __name__ == "__main__":
    input_image = "data/images-test/all_objects/background01-datasheet003-frame1.jpg"
    model_path = "trained_model.h5"
    mapping = annotation_loader.load_mapping()
    annotation = mapping["all_objects\\" + os.path.basename(input_image)]

    print("Loading model...")
    best_model = keras.models.load_model(model_path)

    img = image.load_img(input_image, target_size=(224, 400))
    # We trained with batches, since we only insert one image, we have to add one extra dimension with reshape
    x = np.reshape(image.img_to_array(img), (1, 224, 400, 3))
    y = best_model.predict(x)

    full_image = Image.open(input_image)
    image_draw = ImageDraw(full_image, 'RGBA')
    image_draw.polygon(annotation, fill=(0, 255, 0, 80))
    image_draw.polygon(y.flatten(), fill=(255, 0, 0, 80))
    full_image.show()
    full_image.save("detection.jpg")
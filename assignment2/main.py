import argparse

import dataset_splitter
import ocr_downloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")
    # parser.add_argument("--model_name", type=str, default="res_net_50_gap",
    #                     help="The model used for training the network. Run >python models/ConfigurationFactory.py to get a list of all available configurations")
    # parser.add_argument("--width", default=400, type=int, help="Width of the input-images for the network in pixel")
    # parser.add_argument("--height", default=224, type=int, help="Height of the input-images for the network in pixel")
    # parser.add_argument("--batch_size", default=16, type=int,
    #                     help="The minibatch-size for training. Reduce to 8 or 4 if your graphics card runs out of "
    #                          "memory, but keep high otherwise to speed up training")
    # parser.add_argument("--use_relative_coordinates", dest="use_relative_coordinates",
    #                     action="store_true", help="Specify, if relative coordinates should be used instead of absolute")
    # parser.set_defaults(use_relative_coordinates=False)
    # parser.add_argument("--standardize", dest="standardize",
    #                     action="store_true", help="Specify, if the input images should be standardized or not")
    # parser.set_defaults(standardize=False)
    #
    flags, unparsed = parser.parse_known_args()

    # Download dataset (I AM Database)
    ocr_downloader.get_dataset(flags.dataset_directory)

    # split data into train-test-validation (60-20-20)
    dataset_splitter.split_dataset()

    # binarize images and deskew document (estimate orientation)


    # compute (binary) features for each blob (character)


    # classify features


    # assign labels/label probabilities


    # dictionary correction


    # page_detection_downloader.get_dataset(flags.dataset_directory)
    #
    # # Split dataset into training, validation and test set
    # dataset_splitter.split_dataset()
    #
    # # converts videos from ../data/page-detection-... into images
    # # and saves the frames of each image into the associated directory
    # # ../data/images-.../all objects as *.jpg
    # video_to_image_converter.load_video_to_image_converter(flags.dataset_directory)
    #
    # # train_model creates a model with the training set
    # model_path = train_model.train(flags.dataset_directory, flags.model_name, flags.width, flags.height, flags.batch_size,
    #             flags.use_relative_coordinates, flags.standardize)
    #
    # # predict images
    # predict.predict(flags.dataset_directory,model_path, flags.width, flags.height,flags.use_relative_coordinates, flags.standardize)



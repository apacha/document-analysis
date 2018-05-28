import os
import random
import shutil
import numpy as np

def copy_image_and_annotation(source_directory, destination_directory, file):
    image_source_path = os.path.join(source_directory, file)
    image_target_path = os.path.join(destination_directory, file)
    annotation_source_path = os.path.join(source_directory, file)
    annotation_target_path = os.path.join(destination_directory, file)

    os.makedirs(os.path.dirname(image_target_path), exist_ok=True)
    shutil.copy(image_source_path, image_target_path)
    shutil.copy(annotation_source_path, annotation_target_path)


def split_dataset():
    dataset_directory = "./data/I AM printed"
    training_directory = "./data/I AM printed-training"
    validation_directory = "./data/I AM printed-validation"
    test_directory = "./data/I AM printed-test"

    if os.path.exists(training_directory):
        shutil.rmtree(training_directory)
        os.makedirs(training_directory)
    if os.path.exists(validation_directory):
        shutil.rmtree(validation_directory)
        os.makedirs(validation_directory)
    if os.path.exists(test_directory):
        shutil.rmtree(test_directory)
        os.makedirs(test_directory)

    #remove wrong annotated image
    os.remove(dataset_directory + "/a04-006.png")
    os.remove(dataset_directory + "/a04-006.xml")
    fileIndex = random_List(dataset_directory)

    fileIndexTrain = fileIndex[0:np.math.ceil(len(fileIndex)*0.6/2.)*2]
    fileIndexTest = fileIndex[len(fileIndexTrain):len(fileIndexTrain) + np.math.ceil(len(fileIndex)*0.2/2.)*2]
    fileIndexValid = fileIndex[len(fileIndexTrain)+len(fileIndexTest):len(fileIndex)]

    for i in range (0,len(fileIndexTest)):
        copy_image_and_annotation(dataset_directory,test_directory,fileIndexTest[i])

    for i in range (0,len(fileIndexTrain)):
        copy_image_and_annotation(dataset_directory,training_directory,fileIndexTrain[i])

    for i in range (0,len(fileIndexValid)):
        copy_image_and_annotation(dataset_directory,validation_directory,fileIndexValid[i])



def random_List(path):
    path, dirs, files = os.walk(path).__next__()
    countData = len(files) / 2

    index = np.int_(np.arange(0, countData))
    random.shuffle(index)

    shuffledData = []

    for i in range(0,len(index)):
        shuffledData.append(files[index[i] * 2])
        shuffledData.append(files[index[i] * 2 + 1])

    return shuffledData


if __name__ == "__main__":
    split_dataset()


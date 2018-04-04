import argparse
import os
from distutils import dir_util
from omrdatasettools.downloaders.DatasetDownloader import DatasetDownloader


class PageDetectionDownloader(DatasetDownloader):
    def __init__(self, destination_directory: str):
        """
        Create and initializes a new dataset.
        :param destination_directory: The root directory, into which the data will be copied.
        """
        super().__init__(destination_directory)

    def get_dataset_download_url(self) -> str:
        # If this link does not work anymore, you can download the tar-ball from
        # https://sourceforge.net/projects/openomr/
        # and you will find the images in OpenOMR/neuralnetwork/ + train/validation/test folders
        return "https://cloud.cvl.tuwien.ac.at/s/7k0g6MS4YH3MTRy/download?path=%2F&files=page-detection.zip"

    def get_dataset_filename(self) -> str:
        return "page-detection.zip"

    def download_and_extract_dataset(self):
        if not os.path.exists(self.get_dataset_filename()):
            print("Downloading PageDetection dataset...")
            self.download_file(self.get_dataset_download_url(), self.get_dataset_filename())

        print("Extracting PageDetection dataset...")
        self.extract_dataset(self.destination_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_known_args()

    dataset = PageDetectionDownloader(flags.dataset_directory)
    dataset.download_and_extract_dataset()

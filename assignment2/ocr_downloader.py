import argparse
import os
from distutils import dir_util
from omrdatasettools.downloaders.DatasetDownloader import DatasetDownloader


class IamPrintedDatasetDownloader(DatasetDownloader):
    def __init__(self, destination_directory: str):
        """
        Create and initializes a new dataset.
        :param destination_directory: The root directory, into which the data will be copied.
        """
        super().__init__(destination_directory)

    def get_dataset_download_url(self) -> str:
        return "ftp://scruffy.caa.tuwien.ac.at/staff/diem/DAVU/I%20AM%20printed.zip"

    def get_dataset_filename(self) -> str:
        return "IAM printed.zip"

    def download_and_extract_dataset(self):
        if not os.path.exists(self.get_dataset_filename()):
            print("Downloading OCR dataset...")
            self.download_file(self.get_dataset_download_url(), self.get_dataset_filename())

        print("Extracting OCR dataset...")
        self.extract_dataset(self.destination_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data",
        help="The directory, where the extracted dataset will be copied to")

    flags, unparsed = parser.parse_known_args()

    dataset_downloader = IamPrintedDatasetDownloader(flags.dataset_directory)
    dataset_downloader.download_and_extract_dataset()

# Running the training

To run the training, you first have to prepare the data, but running the following three commands:

```shell
python page_detection_downloader.py
python dataset_splitter.py
python video_to_image_converter.py
```

Then you can start the actual training, by calling

```shell
python train_model.py --use_relative_coordinates --standardize
```
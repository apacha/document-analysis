# Training and running the page detection

To run the full pipeline, simply call `main.py`.

## Running the individual steps

To run the training, you first have to prepare the data, by running the following three commands:

```shell
python page_detection_downloader.py
python dataset_splitter.py
python video_to_image_converter.py
```

Then you can start the actual training, by calling

```shell
python train_model.py --use_relative_coordinates --standardize
```

Once, you have a trained model, you can run inference by calling

```shell
python predict.py --model_path 2018-04-16_res_net_50_gap_400x224_relative_standardize.h5 --use_relative_coordinates --standardize
```

## Pretrained models

If you just want to test the procedure, you can find the pre-trained models for download at

- [ResNet50 GAP Absolute Coordinates](https://owncloud.tuwien.ac.at/index.php/s/Svk8cui4wQjvfvo)
- [ResNet50 GAP Relative Coordinates and Standardization](https://owncloud.tuwien.ac.at/index.php/s/NA1dqaiw8gmg1Bq)
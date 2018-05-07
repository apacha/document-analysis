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
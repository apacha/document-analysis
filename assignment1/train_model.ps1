$pathToGitRoot = "C:/Users/Alex/Repositories/document-analysis/assignment1"
$pathToSourceRoot = "$($pathToGitRoot)"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)\2018-04-19_res_net_50_pyramid_800x448_relative_standardize.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_pyramid --width 800 --height 448 --batch_size 8 --use_relative_coordinates --standardize
Stop-Transcript

exit

Start-Transcript -path "$($pathToTranscript)\2018-04-17_inception_resnet_v2_gap_400x224_relative_standardize.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name inception_resnet_v2_gap --width 400 --height 224 --batch_size 16 --use_relative_coordinates --standardize
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-16_res_net_50_gap_preserve_space_400x224_relative_standardize.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_gap_preserve_space --width 400 --height 224 --batch_size 16 --use_relative_coordinates --standardize
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-16_res_net_50_gap_400x224_relative_standardize.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_gap --width 400 --height 224 --batch_size 16 --use_relative_coordinates --standardize
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-15_res_net_50_gap_960x540_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_gap --width 960 --height 540 --batch_size 4 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-15_xception_gap_960x540_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name xception_gap --width 960 --height 540 --batch_size 4 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-15_inception_resnet_v2_gap_960x540_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name inception_resnet_v2_gap --width 960 --height 540 --batch_size 4 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-13_res_net_50_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50 --width 400 --height 224 --batch_size 16 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-14_xception_gap_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name xception_gap --width 400 --height 224 --batch_size 8 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-14_inception_resnet_v2_gap_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name inception_resnet_v2_gap --width 400 --height 224 --batch_size 8 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-15_res_net_50_gap_1920x1080_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_gap --width 1920 --height 1080 --batch_size 1 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-13_res_net_50_gap_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_gap --width 400 --height 224 --batch_size 8 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-12_res_net_50_2_600x336_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_2 --width 600 --height 336 --batch_size 4 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-12_inception_resnet_v2_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name inception_resnet_v2 --width 400 --height 224 --batch_size 8 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-12_res_net_50_2_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50_2 --width 400 --height 224 --batch_size 16 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-11_res_net_50_400x224_relative-coordinates.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50 --width 400 --height 224 --batch_size 16 --use_relative_coordinates
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-10_dense_net_400x224.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name dense_net --width 400 --height 224 
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-10_res_net_50_800x448.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50 --width 800 --height 448 --batch_size 8
Stop-Transcript

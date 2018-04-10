$pathToGitRoot = "C:/Users/Alex/Repositories/document-analysis/assignment1"
$pathToSourceRoot = "$($pathToGitRoot)"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToSourceRoot

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)\2018-04-10_dense_net_400x224.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name dense_net --width 400 --height 224 
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)\2018-04-10_res_net_50_800x448.txt" -append
python "$($pathToSourceRoot)\train_model.py" --dataset_directory data --model_name res_net_50 --width 800 --height 448 --batch_size 8
Stop-Transcript

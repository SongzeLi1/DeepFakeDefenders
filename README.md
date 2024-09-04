# DeepFakeDefenders

Here is the code of our TEAM-N801 for this [*competition*](https://www.kaggle.com/competitions/multi-ffdi/overview), which finished 12th in the image track and 13th in the video track.

## Requirement
```
pip install -r requirements.txt
```

## Deepfake Image Detection
### Train
```
python FFDI-train.py --batch_size [batch] --device [cuda num] --epochs [epoch num] --train_path [train path] --val_path [val path] --train_txt [traintxt_path] --val_txt [valtxt_path] --learning_rate [learning rate]
```
### infer
```
python FFDI-infer.py --batch_size [batch] --img_file_path [data_path]
```


## Deepfake Audio-Video Detection
### Train
```
python FFDV-train.py [yaml_path]
```
### infer
```
python FFDV-infer.py [yaml_path]
```

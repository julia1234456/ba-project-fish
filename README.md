# ba-project-fish

1) DATASET PREPARATION
 - prepare_framesVal.py & prepare_framesTrain.py :
split the dataset of images in a training set and a validation set (80/20)

- data_cleaning_json.py :
remove the annotations in the json file of the images that contain "false" annotations if the video has not been annotated to the end.
we can't select the annotated frames to be displayed in the json file directly from CVAT and we must remove the unnecessary/uncorrected annotations.

2) DATA TRAINING DETECTRON2

- train_fish.py :
Most important file that allows to run on detectron2 with a custom dataset. 

3) CONFIG FILES
For GTR, better to run on the GTR_FISH_NEW_CONFIG.yaml

4) GTR TRAINING
   
- fish_GTR.py :
File used to train on GTR -> fail (data not aligned in memory) when trying to display the predictions

6) OTHER FILES
Several files to process the data to the correct form. 




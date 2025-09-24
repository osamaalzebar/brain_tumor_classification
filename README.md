# brain_tumor_classification
Improved brain tumor classification model using ensemble and transfer learning Techniques



Part A (Training)
To retrain our model,use the code in the 'our_model' directory  and do the following steps :
1- clone the repo of RAM modelas our model makes use of it. clone it from  "https://github.com/xinyu1205/recognize-anything"  and place it inside 'our_model' directory.
2- download the RAM checkpoint from "https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth"  and place it in the 'checkpoint' folder in the 'our_model' directory .
3-  download the dataset from 'https://www.kaggle.com/datasets/briscdataset/brisc2025'. make sure to download the raw non-augmented data and unzip it
4- use the script named 'dataset_prep' to prepare a data folder cintaining all the MRI slices and create a corresponding .csv files for the labels. To correctly use the dataset_pre script, pass to thhe variable named 'base_dir' inside the script,the local path of the folder in 
your laptop that contains the unzipped files form the dataset original file.
5- train each of the five models by using the scripts whose names start by 'train' (e.g train_densenet). Before runnign the script, pass the coreect paths of the data folder and .csv file that resulted from step 4 
6- to test the model, use the script named 'maj_5'. before running the script, pass to the correct paths of the data folder, .csv file, and the checkpoint paths from step 5. 



Part  B (Testing)
Quick testing:
 Our checkppoints are adirectly available at : "https://huggingface.co/osama-yzu/brain_tumor_checkpoints/tree/main/checkpoints"


 Comparison with other models :
 The remaining folders (other then 'our_model' folder) can verify the results of comparing our model with other models. All training and testiing script require step 3 and 4 form part A  while  the model in the directory '5_model_majority_vote' requires step 1,2,3,4 from part (A). The checkpoints we used for compariosn are also available at : "https://huggingface.co/osama-yzu/brain_tumor_checkpoints/tree/main/checkpoints" with folder names thaat match the folder names in this repository. 

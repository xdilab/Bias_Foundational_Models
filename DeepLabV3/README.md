This project performs automatic brain tumor segmentation on MRI scans using a DeepLabV3+ model with a ResNet-101 backbone.
It processes 2D slices into multi-slice inputs, applies augmentations, trains the model, and evaluates predictions with metrics like Dice, IoU, and F1.
The pipeline outputs trained models, validation results, and visualization of predicted tumor masks.

Requires installation of this datatset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Instructions for MRI_ImageSeg_DeepLabV3_Slices_V2:
Replace files_dir (Found in cell 8) with the path to your downloaded folder titled lgg-mri-segmentation/kaggle_3m
Replace data_df (Found in cell 9) with the path to your data.csv file found in the kaggle_3m folder
Run each cell in order after these paths are replaced

Ex Output:
Epoch 3/150
Train Loss: 0.4512 | Val Loss: 0.3897
IoU: 0.7324 (thr=0.50) | F1: 0.8453
✅ Saved best model at IoU=0.7324, thr=0.50


# BH-LymphDS

Code for the paper "Deep Learning of Histopathological Images Enables Accurate Diagnosis of Lymphoma". This repository mainly contains the source code for lymphoma classification based on deep learning, which is used to extract the features of patch images. Subsequently, the multi-branch clustering-constrained attention multiple instance learning (CLAM) is employed to extract the features at the whole slide image (WSI) level, and the KNN classifier is utilized to diagnose the lymphoma subtypes of slides. The CLAM and the KNN methods can be found in [CLAM](https://github.com/mahmoodlab/CLAM/tree/master)and the Scikit-learn Python Library.



![](Data/BH_LymphDS_overview.png?v=1&type=image)

## Folder Structure

It is necessary to split the WSI files into patch images with a size of 512×512 pixels in advance, and organize these files in the required format. Users are required to save all the patch images cropped from each WSI in the same folder, and there will be as many folders as the number of WSIs. The images are expected to be organized as follows:

```bash
Dataset
   ├──WSI-1
           ├──Patch-1.jpg
           ├──Patch-2.jpg
           ├──Patch-3.jpg
   ├──WSI-2
           ├──Patch-1.jpg
           ├──Patch-2.jpg
           ├──Patch-3.jpg  
```

After the dataset is well organized, it is also necessary to configure the label file, which is in the txt format. These label files contains the path of patch images and corresponding labels. The label files of the training cohort, internal testing cohort and external testing cohort are saved in the same folders named `train.txt`, `val.txt` and `test.txt` respectively. A `labels.txt` also needs to be included in this folder, in which the categories of the labels are listed. We have provided an example of a dataset and placed it in the`\Data\input_images`. We also provided the corresponding label files for reference, and these files are placed in the`\Data\Label_Files`.

## Finetune Model

The deep Learning (DL) functions of the BH-LymphDS are all integrated into main.py. The weight file of backbone model is placed in the `\Data\pretrained_Weight` . Run `main.py --train` to train DL backbone model. The hyperparameters of the model can be modified in the main.py. Further hyperparameters and functions need to be modified in the `\BH_LymphDS\function\run_classification.py`. The output results are defaultly stored in the `\Output\Train`. You can use following command for model training.

```shell
python main.py --train --model_name SwinTransformer --image_data .\Data\input_images --label_dir .\Data\Label_Files --model_path .\Data\pretrained_Weight
```

## Inference

Run `main.py --predict` to predict the lymphoma subtype of patch level images. The hyperparameters of the model can be modified in the main.py file. Further hyperparameters and functions need to be modified in the `\BH_LymphDS\function\run_predict.py`. The output results are defaultly stored in the `\Output\Patch_Predict`. You can use following command for subtypes classification.

```Shell
python main.py --predict --model_name SwinTransformer --image_data .\Data\input_images --label_dir .\Data\Label_Files --model_path .\Data\pretrained_Weight
```

## Extract Patch Level Features

Run `main.py --feature_extract` to extract image features. The parameters need to be modified in the `\function\Feature_Extract.py`. The output results are defaultly stored in the `\Output\Patch_Features`. You can use following command for feature extraction.

```Shell
python main.py --feature_extract --model_name SwinTransformer --image_data .\Data\input_images --label_dir .\Data\Label_Files --model_path .\Data\pretrained_Weight
```

## Visualization

Run `main.py --gradcam` to perform heatmap visualization with GradCAM method. The parameters need to be modified in the `\function\GradCam.py`. The output results are defaultly stored in the `\Output\GradCam`. You can use following command for GradCAM visualization.

```Shell
python main.py --gradcam --model_name SwinTransformer --image_data .\Data\input_images --label_dir .\Data\Label_Files --model_path .\Data\pretrained_Weight
```

## Reference and Acknowledgements

We thank the authors and developers for their contribution as below.

[MMLAB](https://github.com/open-mmlab/mmpretrain)

[CLAM](https://github.com/mahmoodlab/CLAM/tree/master)

[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

[OneKey](https://github.com/OnekeyAI-Platform/onekey)

## License

This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

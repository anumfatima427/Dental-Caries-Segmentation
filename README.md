
# :bell: Mask-RCNN Based Periodontitis Detection

- This repository represents "Dental Lesion Detection Using Deep learning.
- With the help of this project, we can detect dental lesions using periapical radiographs.

### :page_with_curl: Description

- In this project, I have used [deep learning technique] for dental disease segmentation. 
- The code is executed on Google Colab and the dataset and files are saved on Google Drive.


### :pencil: Requirements
- Python
- Tensorflow


### :computer: Dataset
[Periapical Radiograph Dataset with Annotations](https://drive.google.com/drive/folders/1zjl4sF3-s8z1yRnSDYdVLXaVDOyPwdvm?usp=sharing)
- The dataset contains 516 images of periodontal lesions i.e., periodontal lesions i.e., 'PrimaryEndodontic', 'PrimaryEndowithSecondaryPerio', 'PrimaryPeriodontal', 'PrimaryPeriowithSecondaryEndo', 'TrueCombined'. 
- The annotations for these images is generated using VIA Image Annotators and are saved in json format in via_annotation.json file. 

![Dataset with Mask Images](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Installation
To run Mask-RCNN on Google Colab, you will need python version 3.5 with tensorflow(1.14.0), keras(2.0.8) and h5py(2.10.0)

```bash
!apt-get install python3.5
!pip install tensorflow==1.14.0
!pip install keras==2.0.8
!pip install h5py==2.10.0
```

## :pencil2: Notebook Organization 

1. [dataset.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/dataset.ipynb) to load the dataset comprising images of five different periodontal lesions i.e., 'PrimaryEndodontic', 'PrimaryEndowithSecondaryPerio', 'PrimaryPeriodontal', 'PrimaryPeriowithSecondaryEndo', 'TrueCombined'.
2. [generate_annotations.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/generate_annotations.ipynb) to generate annotations using via_annotation.json file for train and val folder for five different periodontal lesions.
3. [train_model.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/train_model.ipynb) to train the model using pre-trained model and save the weights in logs folder.
4. [evaluate_model.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/evaluate_model.ipynb)to evaluate rhe performance of mrcnn.
5. [Testing RCNN.ipynb] to test the performance of the model on test images.

## :pencil2: Folder/File Structure

```bash

├── dataset                                   # place the dataset here            
│       ├── train
│       │   ├── <image_file 1>                # accept .JPG or jpg
│       │   ├── <image_file 2>
│       │   ├── ...
│       │   └── via_export_json.json          # corresponded single annotation file, must be named like this
│       ├── val      
│       │   ├── <image_file 1>                # accept .jpg or .jpeg file
│       │   ├── <image_file 2>
│       │   ├── ...
│       │   └── via_export_json.json         #make sure the annotations are saved in both folders with same name
├── logs                                      # log folder
│   ├── saved model.h5               

├── mrcnn                                     # model folder
├── test                                      # test folder
│   ├── images

├── 
├── dataset.ipynb                                # dataset configuration
├── generate_annotations.ipynb                   #generate annotations using json file and images
├── evaluation.ipynb                             # weight evaluation
├── train_model.ipynb                            # training the model using pretrained weights
├── testing_model.ipynb                          #test to see model's performance on dental radiographs

└── README.md                            
```
## :notebook_with_decorative_cover: Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

    
## Helpful Resources

 - [Practical Implementation Guide](https://www.youtube.com/watch?v=1u-dm5JMH1Q&t=2s&ab_channel=CodeWithAarohi)
 - [Tune Hyperparameters](https://medium.com/analytics-vidhya/taming-the-hyper-parameters-of-mask-rcnn-3742cb3f0e1b)
 - [Helpful Blog](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a) to learn how r-cnn works!
 - [MRCNN code for video/image object detection](https://github.com/quanghuy0497/Mask_R-CNN)


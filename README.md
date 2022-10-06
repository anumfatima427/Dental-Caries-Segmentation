
# :bell: Deep Learning-Based Multiclass Instance Segmentation for Dental Lesions Using M-RCNN

- This repository represents "Dental Lesion Detection Using Deep learning.
- With the help of this project, we can detect dental lesions using periapical radiographs.

### :page_with_curl: Description

- In this project, I have used M-RCNN for multiclass instance segmentation. 
- The code is executed on Google Colab and the dataset and files are saved on Google Drive.


### :pencil: Requirements
- Python
- Tensorflow
- Keras


### :computer: Dataset
[Periapical Radiograph Dataset with Annotations](https://drive.google.com/drive/folders/1mUb_U4cJA_UNzXO6tySE9YlKMM2DV13Q)
- The dataset contains 516 images of periodontal lesions i.e., periodontal lesions i.e., 'PrimaryEndodontic', 'PrimaryEndowithSecondaryPerio', 'PrimaryPeriodontal', 'PrimaryPeriowithSecondaryEndo', 'TrueCombined'. 
- The annotations for these images is generated using VIA Image Annotators and are saved in json format in via_annotation.json file. To view json files, click [here](http://jsonviewer.stack.hu/)

#### Image Preprocessing 
The images in the dataset are preprocessed usign CLAHE, the results are shown below
![Fig  6 Preprocessed Image](https://user-images.githubusercontent.com/66737416/194244705-157c1812-a3e2-4b92-b2d1-fb7f0e5603aa.png)

#### Overlaid Image Masks
![Fig  7 Mask overlaid on original image](https://user-images.githubusercontent.com/66737416/194244882-7dc16657-020f-498b-9e14-5a046601a7f8.png)


## Installation
To run Mask-RCNN on Google Colab, you will need python version 3.5 with tensorflow(1.14.0), keras(2.0.8) and h5py(2.10.0)

```bash
!apt-get install python3.5
!pip install tensorflow==1.14.0
!pip install keras==2.2.5
!pip install h5py==2.10.0
```

## :pencil2: Notebook Organization 

1. [dataset.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/dataset.ipynb) to load the dataset comprising images of five different periodontal lesions i.e., 'PrimaryEndodontic', 'PrimaryEndowithSecondaryPerio', 'PrimaryPeriodontal', 'PrimaryPeriowithSecondaryEndo', 'TrueCombined'.
2. [Generated_Masks_from_VIA_Annotations.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/Generated_Masks_from_VIA_Annotations.ipynb) to generate annotations using via_annotation.json file for train and val folder for five different periodontal lesions.
3. [train_model(5_cross_validation).py](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/train_model(5_cross_validation).py) to train the model using pre-trained model and save the weights in logs folder. This model was trained was 7 epochs and weights are stored [here](https://drive.google.com/drive/folders/1-B6HoGZ0Rl27k77EUWme7OD0ufpttniU)
4. [evaluate_model.ipynb](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/evaluate_model.ipynb)to evaluate rhe performance of mrcnn.
5. [test_model.py](https://github.com/anumfatima427/Dental-Caries-Segmentation/blob/main/test_model.py) to test the performance of the model on test images.

## Folder/File Structure

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
├── images                                      # test folder
│   ├── test image

├── 
├── dataset.ipynb                                # dataset configuration
├── Generated_Masks_from_VIA_Annotations.ipynb.ipynb    #generate annotations using json file and images
├── evaluation.ipynb                             # weight evaluation
├── train_model(5-cross validation).py           # training the model using pretrained weights
├── test_model.py                               #test to see model's performance on dental radiographs

└── README.md                            
```


## Json Annotation Format

```bash
{ 'filename': '<image_name>.jpg',
           'regions': {
               '0': {
                   'region_attributes': {},
                   'shape_attributes': {
                       'all_points_x': [...],
                       'all_points_y': [...],
                       'name': <class_name>}},
               ... more regions ...
           },
           'size': <image_size>
}
```

## :notebook_with_decorative_cover: Proposed Model 
The process flow of the proposed dental lesion detection is shown below. First, the
collected annotated images are preprocessed to remove noise, enhance contrast and improve resolution of the images. Next, the preprocessed images are used by the proposed
lightweight backbone network for feature extraction, the extracted feature maps are then
forwarded to the region proposal network (RPN) that generates region proposals using
the feature maps and forwards it to the ROI align block, this block processes both the
feature maps and region proposals and classifies the input image using fully connected
layers. The model further exhibits the bounding box on the identified region so it can
be visualized.

![Fig  16 Proposed process flow](https://user-images.githubusercontent.com/66737416/194245393-e953c71a-02cf-4913-85b2-b0d155587fe1.png)

The training process of M-RCNN requires high-performance computing resources to
learn and analyze substantial information obtained from medical imagery. To reduce the
performance requirement of M-RCNN and ensure that it operates properly, a lightweight
backbone network is utilzied with M-RCNN to classify five types of endo-perio lesions.
The focus of this research is to propose a lightweight M-RCNN model that can operate
on platforms with less computational resources such as graphic process unit (GPU) and
memory and provide performance similar to that of the original M-RCNN. 

For this purpose, a lightweight network MobileNet-v3 is utilized for feature extraction
followed by a depthwise separable convolutional layer proceeding tiny region proposal
network (RPN) to extract candidate regions with potential targets [55]. The RPN
generates anchor boxes for each classified object using the softmax activation function.
The extracted proposal regions along with feature maps are applied to ROI alignment to
locate all the feature map areas. ROI alignment wraps different feature vectors which are
then applied to mask generation and classification. The fully connected layer provides
classification and bounding boxes for each identified endo-perio lesion. The masks are
generated by the convolution layer for each object at the pixel level. The proposed
framework for lightweight M-RCNN for dental lesion classification and localization is
depicted below: 

![Fig  3 Architecture of Lightweight M-RCNN](https://user-images.githubusercontent.com/66737416/194245677-b802bff8-beee-4600-a224-328709e5715a.jpg)


### Proposed Model Result
- A lightweight network MobileNet-v3 is utilized for feature extraction followed by a depthwise separable convolutional layer proceeding tiny region proposal network (RPN) to extract candidate regions with potential targets. 
- The RPN generates anchor boxes for each classified object using the softmax activation function.
- The extracted proposal regions along with feature maps are applied to ROI alignment to locate all the feature map areas. ROI alignment wraps different feature vectors which are then applied to mask generation and classification. 
- The fully connected layer provides classification and bounding boxes for each identified endo-perio lesion. 
- The masks are generated by the convolution layer for each object at the pixel level.

![Review Paper (17)](https://user-images.githubusercontent.com/66737416/194244117-6d1eba13-f438-4106-8f14-6fcd9f7e35e5.png)


## Helpful Resources

 - [Practical Implementation Guide](https://www.youtube.com/watch?v=1u-dm5JMH1Q&t=2s&ab_channel=CodeWithAarohi)
 - [Tune Hyperparameters](https://medium.com/analytics-vidhya/taming-the-hyper-parameters-of-mask-rcnn-3742cb3f0e1b)
 - [Helpful Blog](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a) to learn how r-cnn works!
 - [MRCNN code for video/image object detection](https://github.com/quanghuy0497/Mask_R-CNN)

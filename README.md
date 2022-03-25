# 110_MedicalAI_Lab1_Pneumonia_Detection
## Introduction

   According to the World Health Organization (WHO), pneumonia kills about 2 million children under 5 years old every year and is consistently estimated as the single leading cause of childhood mortality.
    
   Therefore, accurate and timely diagnosis is imperative. One key element of diagnosis is radiographic data, since chest X-rays are routinely obtained as standard of care and can help differentiate between different types of pneumonia. However, rapid radiologic interpretation of images is not always available, particularly in the low-resource settings where childhood pneumonia has the highest incidence and highest rates of mortality. 
   
   We access the pneumonia dataset from kaggle competition(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) which mentioned in the class before. the datset including training set, validation set and testing set. we only use train set as our training data which contains 5216 chest X-ray images from children, including 3875 characterized as depicting pneumonia and 1341 normal, to train the CNN system. Then, we test the model with 234 normal images and 390 pneumonia images (242 bacterial and 148 viral) from 624 patients.
   
   This experiment aims is to: (1) Write our own custom dataloader, (2) Classify Pneumonia with custom own model, (3) Using ResNet50, ResNet101 and ResNet152 to analysis pneumonia from chest x ray image and (4) Compare the experiment results between different models (5) Optimize classfication accuracy > 90%.

---
## Environment
* Framework: Pytorch
* Library versions: See `requirement.txt`

---
## Usage
### Training
`python3 train.py`
### Testing
`python3 inference.py`

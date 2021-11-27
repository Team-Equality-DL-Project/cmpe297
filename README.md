# CMPE 297 - Advanced Deep Learning Project Repository

## Alzheimer's - Medical Image Analysis using SimCLR

### Project overview
Alzheimer's disease is a neurodegenerative disease primarily affecting middle-aged and elderly people. Medical imaging techniques are vastly used for primary diagnosis and a lot of computer aided algorithms assist physicians in analysing the brain images. Deep learning based techniques are becoming increasingly adopted in the medical field for detection and study of diseases. The challenging part of creating deep learning models is the lack of high quality labelled dataset. Such conditions are ideally suited for self-supervised or semi-supervised deep learning. In this project we used the self-supervised learning, **SimCLR** (A Simple Framework for Contrastive Learning of Visual Representations) as well as the semi-supervised learning, **SimCLRv2** and **Teacher-Student Knowledge Distillation** to classify Alzheimer's MRI images into four classes based on severity. 


### Dataset
The Alzheimer dataset is sourced from Kaggle and consists of MRI images for four different stages of dementia (none, very mild, mild, moderate).The training dataset has 5,121 images and the testing dataset has 1,279 images. The figure shows sample images from each of the four classes of MRI images.

![image](https://user-images.githubusercontent.com/70080503/143667524-1a68b7de-de06-415f-934c-8a52a5dcbf51.png)
### Methodologies
  #### SimCLR
  SimCLR is a self-supervised learning technique. It learns representations by maximizing agreement between differently augmented views of the same data example    via a contrastive loss in the latent space. It has four components: the data augmentation module, base encoder, the projection head and the Contrastive Loss  Function. 

  #### SimCLRv2
  SimCLRv2 is the semi-supervised version of SimCLR. SimCLRv2 can be summarized in three steps. First step is the task agnostic unsupervised pre-training for   learning useful representations. The second step is supervised fine-tuning on labeled samples for a specific task. The final step is the self-training or distillation with task-specific unlabeled examples. 

  #### Teacher-Student Knowledge Distillation
  Knowledge distillation is an intuitive way of improving the performance of deep learning models on thin computational devices like mobiles. First, a teacher network is trained on huge amounts of data to learn the features. Then the student is trained to mimic teachers output instead of training on the raw data directly.

![image](https://user-images.githubusercontent.com/70080503/143666863-ed6a873f-1376-4580-bb13-2755578bf6f0.png)

### Model Development
Configuration 
- **SimCLRv1 - Pre-trained on imagenet with Alzheimers images**
    - Extract image features using Resnet50 using pretrained weights and bias from Imagenet.
    - Create a one layer linear model with 128 features matching the last layer of the Resnet50 model.
    - Set the activation to be softmax with loss as cross-entropy loss.
    - Use a linear model for training and prediction with a 64 batch size and early stopping integrated.

- **SimCLRv2 - Fine-tune a pre-trained SimCLRv2 model on Alzheimer's images**
    - Load a pre-trained SimCLR model on 100% of labels from the hub module (gs://simclr-checkpoints/simclrv2/finetuned_100pct).
    - Attach a trainable linear layer with LARS optimization (weight decay = 1e-3, momentum = 0.8, learning rate = cosine decay) and a batch size of 32. 

- **SimCLRv2 - Perform Student Teacher knowledge distillation using fine-tuned simClrv2**
    - Load a pre-trained SimCLR model on 100% of labels from the hub module (gs://simclr-checkpoints/simclrv2/finetuned_100pct).
    - Fine-tune the pre-trained SimCLR model on Alzheimer images to create a teacher model.
    - Build a student model with 3 convolutional layers, batch normalization, relu activation layer, max pooling layers, and 2 dense layers for prediction.
    - Train only the student model with the teacher’s label with LARS optimization (weight decay = 1e-3, momentum = 0.8, learning rate = cosine decay) and softmax cross-entropy with temperature with a batch size of 16.


### Prediction Results
![image](https://user-images.githubusercontent.com/70080503/143667506-7cd31c3e-132b-4f37-90cd-cfd0f86705f4.png)
### Model comparison
From the ROC-AUC curves comparison, we observed that SimCLRv1 performed the best with the highest micro-average (0.82) and macro-average (0.68) numbers. SimCLRv1 also had the best micro-average (0.60) on the Precision-Recall curve. Across all the Confusion Matrices, we also observed that the incorrectly classified images for each model were skewed towards non demented or very mild demented. This aligned well with our initial concern of working with a highly imbalanced dataset where there are more images for these two classes than the rest of the classes.
![image](https://user-images.githubusercontent.com/70080503/143667488-5f531f9c-aab4-41c6-bb33-22b7ffbb66f4.png)

### Model deployment
![image](https://user-images.githubusercontent.com/70080503/143667677-915d089a-5f67-44d9-93b6-e9e910cedba6.png)

#### GCP App Engine deployment of the model endpoint
Our application is live at: https://alzheimers-331518.uc.r.appspot.com/

![image](https://user-images.githubusercontent.com/70080503/143667394-b53c3797-74db-4027-87e3-92636f155e90.png)

### Deliverables

- [Project report](https://github.com/Team-Equality-DL-Project/cmpe297/blob/main/docs/CMPE297_Project%20Report.pdf)
- [Presentation](https://github.com/Team-Equality-DL-Project/cmpe297/blob/main/docs/CMPE297_Project_Presentation.pdf)
- [Colabs](https://github.com/Team-Equality-DL-Project/cmpe297/tree/main/notebooks)
- [TFX pipeline](https://github.com/Team-Equality-DL-Project/cmpe297/blob/main/notebooks/tfx_pipeline.ipynb)


### Team Equality members

| Team members  | Contribution                                       |
|---------------|----------------------------------------------------|
| Abhishek Bais |                                                    |
| Haley Feng    |                                                    |
| Princy Joy    | TFX pipeline, Flask App, GCP App engine deployment |
| Shannon Phu   |                                                    |



# Smart-Agriculture-AI-driven-Crop-Disease-Detection-using-Leaf-Images-

### Project Overview
Crop diseases are a major threat to agricultural productivity. Early detection and treatment of these diseases are crucial for preventing significant crop loss. This project involves developing a deep learning model to detect and classify crop diseases from leaf images.

### Project Goals
#### Data Collection and Preprocessing: Gather a dataset of leaf images showing healthy and diseased plants, preprocess the images.
#### Model Development: Create a convolutional neural network (CNN) to classify different crop diseases.
#### Model Training: Train the model on a labeled dataset of leaf images.
#### Model Evaluation: Evaluate the model's performance using appropriate metrics.
##### Deployment: Develop a web application to upload leaf images and get disease diagnoses.

### Steps for Implementation
#### 1. Data Collection
Use publicly available datasets such as the PlantVillage dataset which contains images of leaves from various crops and associated diseases.

#### 2. Data Preprocessing
##### Normalization: Normalize pixel values to a range of 0 to 1.
##### Resizing: Resize images to a consistent size (e.g., 224x224).
##### Data Augmentation: Apply random transformations like rotations, flips, and zooms to increase the diversity of the training set.
#### 3. Model Development
Develop a CNN using TensorFlow and Keras.

#### 4. Model Training
Split the dataset into training and validation sets, then train the model.

#### 5. Model Evaluation
Evaluate the model using metrics like accuracy, precision, recall, and F1 score.

#### 6. Deployment
Deploy the model using Flask for the backend and a simple HTML/CSS frontend.

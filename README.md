# Mini-Project-5: Solar Guard-Intelligent Defect Detection on Solar Panels using Deep Learning

## Overview
This project focuses on automatically classifying the condition of solar panels using Deep Learning techniques. The model identifies six different panel conditions, including environmental contamination and physical damage, to assist in monitoring the health of solar panels.

The dataset consists of labeled solar panel images representing different panel conditions. After performing image preprocessing and data augmentation, a Convolutional Neural Network (CNN) model is trained to classify the images. An interactive Streamlit web application is developed to allow users to upload a solar panel image and predict its condition along with the confidence score.

## Technologies Used
- **Python** (NumPy): For image preprocessing and numerical computations.
- **TensorFlow & Keras**: For building, training, and evaluating the CNN model.
- **Scikit-learn**: For generating the classification report and confusion matrix.
- **Matplotlib & Seaborn**: For image visualization and model evaluation.
- **Pillow (PIL)**: For image loading and preprocessing.
- **Streamlit**: For developing an interactive web application.

## Steps Involved

### 1. Dataset Preparation
- Loaded the solar panel image dataset from the local directory.
- Verified the dataset structure and image availability.
- Visualized sample images from each class for inspection.
- Split the dataset into training and validation datasets.

### 2. Data Preprocessing
- Resized all images to **224 × 224** pixels.
- Normalized pixel values to improve model performance.
- Applied data augmentation techniques such as random flip, rotation, and zoom.
- Optimized the dataset using cache and prefetch for faster training.

### 3. CNN Model Development
- Built a Convolutional Neural Network (CNN) for multi-class image classification.
- Used multiple Convolution and Max Pooling layers to extract image features.
- Added Dense and Dropout layers to improve learning and reduce overfitting.
- Used a Softmax output layer to classify images into six categories.

### 4. Model Training & Evaluation
- Trained the CNN model using the Adam optimizer and Categorical Crossentropy loss function.
- Applied Early Stopping to reduce overfitting and restore the best model weights.
- Evaluated the model using Accuracy, Precision, Recall, Classification Report, and Confusion Matrix.

### 5. Streamlit Application Development
- Built an interactive web application using Streamlit.
- Allowed users to upload a solar panel image.
- Predicted the condition of the uploaded solar panel image.
- Displayed the predicted condition along with the confidence score through a simple and user-friendly interface.

## Dataset Classes
- Bird-drop
- Clean
- Dusty
- Electrical-damage
- Physical-Damage
- Snow-Covered

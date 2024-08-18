# age_and_gender_detection

## Project Overview

This project implements a system that detects faces in an image and predicts the age and gender of each detected face. The project uses pre-trained deep learning models from OpenCV's DNN module to perform these tasks. The output is an image with annotated bounding boxes around faces, labeled with the predicted age and gender.

## Introduction
Face detection and attribute prediction, such as age and gender, are important applications in computer vision. This project demonstrates how to use deep learning models in OpenCV to detect faces and predict the age and gender of individuals in an image. The solution involves detecting faces first, then predicting age and gender using separate models for each task.

## Prerequisites
Before running the code, ensure that you have the following installed and configured:

+ Python 3.x
+ OpenCV (cv2)
+ Google Colab (optional, for running in a Jupyter environment)

You should also have the following model files:

+ opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb for face detection.
+ age_deploy.prototxt and age_net.caffemodel for age prediction.
+ gender_deploy.prototxt and gender_net.caffemodel for gender prediction.
## Project Structure
The project consists of the following main components:

1. Model Files: Pre-trained models for face detection, age prediction, and gender prediction.
2. Python Script: The main script that loads the models, processes an image, and displays the results.
## Detailed Explanation
### getFaceBox Function
Purpose:

The getFaceBox function detects faces in an input image and returns the image with bounding boxes drawn around detected faces, along with the coordinates of these boxes.

Parameters:

+ net: The pre-trained DNN model for face detection.
+ frame: The input image in which faces are to be detected.
+ conf_threshold: The minimum confidence score to consider a detection as valid (default is 0.7).

**Process**:

1. Copy the Input Frame:

The function starts by creating a copy of the input frame to avoid modifying the original image.
2. Prepare the Image Blob:

The image is preprocessed into a 4D blob, which is the input format required by the DNN model. This blob is scaled and resized to the appropriate dimensions (300x300 pixels), with mean subtraction for color normalization.

3. Perform Face Detection:

The blob is passed through the DNN model, which outputs detections. Each detection includes a confidence score and the coordinates of the bounding box.

4. Filter Detections by Confidence:

The function filters out detections with confidence scores below the threshold. For valid detections, it calculates the bounding box coordinates relative to the original image dimensions.

5. Draw Bounding Boxes:

Bounding boxes are drawn around the detected faces on the copied frame using the calculated coordinates.

6. Return Values:

The function returns the modified frame with bounding boxes and the list of bounding box coordinates.

## age_gender_detector Function
**Purpose**:

The age_gender_detector function takes an input image, detects faces, and then predicts the age and gender for each detected face. The output is an image annotated with the detected faces' age and gender labels.

**Parameters**:
 + frame: The input image containing faces.
**Process**:
1. Face Detection:

The function begins by calling getFaceBox to detect faces in the input image and obtain the bounding boxes.

2. Loop Through Detected Faces:

For each detected face, the function extracts a region of interest (ROI) from the original image. The ROI is expanded with a padding to include some background around the face, which helps improve prediction accuracy.

3. Preprocess ROI:

The face ROI is preprocessed into a blob suitable for the age and gender prediction models. This blob is resized to 227x227 pixels and normalized using predefined mean values.

4. Gender Prediction:

The preprocessed blob is passed through the gender prediction model, and the model outputs a probability distribution over the two possible genders (male and female). The gender with the highest probability is selected as the predicted gender.

5. Age Prediction:

The same blob is passed through the age prediction model, which outputs a probability distribution over predefined age categories. The category with the highest probability is selected as the predicted age group.

6. Annotate Image:

The predicted age and gender are combined into a label, which is placed on the image near the corresponding face.

7. Return Value:

The function returns the annotated image with the age and gender predictions.
##Execution Flow
1. Load Image:

The script starts by loading an input image from the specified file path.

2. Detect and Predict:

The age_gender_detector function is called to process the image, detect faces, and predict age and gender.

3. Display Output:

The annotated image is displayed using the cv2_imshow function, which is specifically used for image display in Google Colab.

## Models Used
### Face Detection Model
Files:
 + opencv_face_detector.pbtxt: Configuration file containing model architecture.
 + opencv_face_detector_uint8.pb: Pre-trained weights file.
Purpose:
 + Detects faces in an input image.
### Age Prediction Model
Files:
 + age_deploy.prototxt: Configuration file for model architecture.
 + age_net.caffemodel: Pre-trained weights file.
Purpose:
 + Predicts the age of a detected face, categorizing it into one of the predefined age groups.
### Gender Prediction Model
Files:
 + gender_deploy.prototxt: Configuration file for model architecture.
 + gender_net.caffemodel: Pre-trained weights file.
Purpose:
 + Predicts the gender of a detected face, outputting either 'Male' or 'Female'.
## Usage Instructions
1. Setup Environment:

Ensure you have all required dependencies installed and model files available at the specified paths.

2. Load and Run Script:

Execute the script in a Python environment, such as Google Colab. Ensure the input image path is correct.

3. View Results:

After running the script, the output image with annotated age and gender predictions will be displayed.

## Conclusion
This project showcases how to leverage deep learning models in OpenCV for practical applications like age and gender prediction. The combination of face detection with attribute prediction has broad applications, including security systems, demographic analysis, and personalized marketing.

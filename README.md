YOLOv8 Custom Model Training and Inference README

This README provides an overview of the code you've uploaded to GitHub, which is related to training a custom YOLOv8 model for object detection. The code seems to be designed for use in Google Colab. Below are explanations of the major parts of the code and how to use it.


Overview:
This code is intended for training a custom YOLOv8 object detection model. It performs tasks such as data preparation, model training, and inference. Here's an overview of the main components:


Installation:
Before running the code, you need to install the Ultralytics library by executing the following command:

!pip install ultralytics

Also, it installs or upgrades the tqdm library:

!pip install tqdm --upgrade


Data Preparation:
∙The code starts by mounting your Google Drive, which is where your data and model will be stored.
∙It defines paths for training and validation data (train_path_img, train_path_label, val_path_img, val_path_label) and a test path (test_path).
∙The train_test_split function is used to split the data into training and testing sets. It also copies image and label files to appropriate directories for training and validation.


Model Training:
∙It checks for Ultralytics library dependencies using ultralytics.checks().
∙The YOLOv8 model is trained using the !yolo command with various parameters, including the model type (model=yolov8s.pt), dataset configuration (data=/content/drive/MyDrive/yolov8/dataset.yaml), and training settings (epochs=10, imgsz=640, batch=8, project=/content/drive/MyDrive/yolov8/training_results, name=football).


Model Inference:
∙The code performs object detection on sample images using the trained model. It uses the !yolo command with different modes (e.g., mode=predict) and specifies the model and source image. Predictions are saved in the /content/runs/detect/ directory.
∙The !cp command is used to copy the prediction results to your Google Drive.


Exporting the Model:
∙The code also exports the trained model in ONNX format using the !yolo command with mode=export. The resulting model can be used for inference in various environments.


Usage:
To use this code for your custom YOLOv8 model training and inference, follow these steps:

1. Install the required libraries at the beginning of your Colab notebook.
2. Mount your Google Drive to access your data and save the results.
3. Configure the data paths and split your dataset using the train_test_split function.
4. Train your YOLOv8 model by adjusting the training parameters and dataset configuration in the !yolo command.
5. Perform object detection on images or videos by using the !yolo command in the inference section.
6. Export the trained model in ONNX format if needed.

Remember to modify the paths and settings according to your specific dataset and project requirements.# CustomYolo

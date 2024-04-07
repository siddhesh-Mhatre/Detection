# Dog Skin Diseases Detection and Identification Using Convolutional Neural Networks

## Introduction
This project aims to classify different skin conditions using deep learning techniques implemented with TensorFlow and Keras. The dataset used for training consists of images of various skin conditions.

## Environment Setup
- This project was developed using Google Colab, a cloud-based Jupyter notebook environment.
- TensorFlow version 2.7.0 was used for deep learning operations. It was installed using the `pip` package manager.
- The project also utilizes other common Python libraries such as `numpy`, `matplotlib`, and `PIL`.

## Data Preparation
- The dataset is stored in the directory `/content/drive/MyDrive/NewSkinData`. It contains images of different skin conditions categorized into subdirectories.
- The `image_dataset_from_directory` function from TensorFlow's `tf.keras.utils` module is used to create training and validation datasets from the directory structure.
- Images are resized to a uniform size of 180x180 pixels to ensure consistency.
- ![image](https://github.com/siddhesh-Mhatre/Detection/assets/80941193/07e53e2d-5b89-43c2-9e81-07aeb7c10be4)


## Model Building
- A Convolutional Neural Network (CNN) architecture is used for image classification. The model consists of convolutional layers followed by max-pooling layers for feature extraction and downsampling.
- The final layers include fully connected (dense) layers for classification, with the number of units equal to the number of skin condition classes.
- The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function.

## Data Augmentation
- Data augmentation is performed using TensorFlow's `Sequential` API to generate additional training data by applying random transformations such as flips, rotations, and zooms to the images.

## Training
- The model is trained on the training dataset for multiple epochs, with validation performed on a separate validation dataset.
- Training progress is monitored using metrics such as accuracy and loss, which are plotted using `matplotlib` to visualize training and validation performance over epochs.

## Model Evaluation
- After training, the model's performance is evaluated on the validation dataset to assess its accuracy and generalization capability.
- The model's predictions on sample images are also visualized using `matplotlib` to demonstrate its classification capability.
- ![image](https://github.com/siddhesh-Mhatre/Detection/assets/80941193/3c0d93e9-ed7a-45c9-93b7-5b014ded4398)

## Expected Output
- The output of the project includes visualizations of training and validation accuracy/loss curves, as well as sample images with predicted skin conditions.

- ![image](https://github.com/siddhesh-Mhatre/Detection/assets/80941193/23b05e23-37cf-4cca-8673-032b0ba5d688)



## Conclusion
- This project demonstrates the application of deep learning techniques for skin condition classification, providing a foundation for further research and development in medical image analysis.
  
## Research Paper
- https://link.springer.com/article/10.1007/s42979-022-01645-5#:~:text=In%20our%20research%2C%20we%20implemented,efficiency%20in%20identifying%20skin%20diseases.

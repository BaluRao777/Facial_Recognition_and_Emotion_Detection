import numpy as np
import cv2


def preprocess_image(image):
    """
    Preprocess a single image: resize and normalize.
    """
    image = cv2.resize( image, (128, 128) )  # Resize the image to 128x128
    image = image.astype( 'float32' ) / 255.0  # Normalize pixel values to the range [0, 1]
    return image


def preprocess_batch(images):
    """
    Preprocess a batch of images.
    """
    return np.array( [preprocess_image( image ) for image in images] )


def plot_history(history):
    """
    Plot training and validation accuracy and loss curves.
    """
    import matplotlib.pyplot as plt

    # Accuracy
    plt.plot( history.history['face_output_accuracy'], label='Face Recognition Accuracy' )
    plt.plot( history.history['emotion_output_accuracy'], label='Emotion Detection Accuracy' )
    plt.title( 'Model Accuracy' )
    plt.ylabel( 'Accuracy' )
    plt.xlabel( 'Epoch' )
    plt.legend( loc='upper left' )
    plt.show()

    # Loss
    plt.plot( history.history['face_output_loss'], label='Face Recognition Loss' )
    plt.plot( history.history['emotion_output_loss'], label='Emotion Detection Loss' )
    plt.title( 'Model Loss' )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epoch' )
    plt.legend( loc='upper left' )
    plt.show()

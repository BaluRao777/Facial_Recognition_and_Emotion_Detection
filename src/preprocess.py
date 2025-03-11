import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(directory):
    images = []
    labels = []

    for label_dir in os.listdir( directory ):
        if os.path.isdir( os.path.join( directory, label_dir ) ):  # Check if it's a folder
            for image_file in os.listdir( os.path.join( directory, label_dir ) ):
                image_path = os.path.join( directory, label_dir, image_file )
                image = load_img( image_path, target_size=(128, 128) )  # Resize to your desired size
                image = img_to_array( image )
                images.append( image )
                labels.append( label_dir )  # Folder name is the label

    images = np.array( images )
    labels = np.array( labels )

    print( f"Loaded {len( images )} images from {directory}" )
    return images, labels


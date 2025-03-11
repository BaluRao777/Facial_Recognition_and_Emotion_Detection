import numpy as np
from keras.models import load_model
from src.preprocess import load_data
from src.utils import preprocess_batch

def evaluate_model():
    # Load the test datasets (assuming they are structured similarly to the training datasets)
    face_test_data, face_test_labels = load_data('data/face_dataset_test')
    emotion_test_data, emotion_test_labels = load_data('data/emotion_dataset_test')

    # Preprocess the test data
    face_test_data = preprocess_batch(face_test_data)
    emotion_test_data = preprocess_batch(emotion_test_data)

    # Load the trained model
    model = load_model('checkpoints/facial_emotion_recognition_model.h5')

    # Evaluate the model
    results = model.evaluate(
        [face_test_data, emotion_test_data],
        {'face_output': face_test_labels, 'emotion_output': emotion_test_labels}
    )

    print("Test Loss and Accuracy:", results)

    return results

if __name__ == "__main__":
    evaluate_model()

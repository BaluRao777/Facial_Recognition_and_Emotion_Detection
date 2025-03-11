import os
import numpy as np
from keras.utils import to_categorical
from src.preprocess import load_data
from src.model import build_model
from sklearn.preprocessing import LabelEncoder
import pickle

def filter_datasets(face_data, face_labels, emotion_data, emotion_labels):
    # Assuming face_labels and emotion_labels are indexed the same way
    # You will filter the emotion data to match the face data size
    emotion_data_filtered = emotion_data[:len(face_data)]
    emotion_labels_filtered = emotion_labels[:len(face_labels)]
    return face_data, face_labels, emotion_data_filtered, emotion_labels_filtered
def train_model():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    # Load face and emotion datasets
    face_data, face_labels = load_data('/Users/kowshikbala/PycharmProjects/Facial Recognition and Emotion Detection Using Deep Learning/data/face_dataset')
    emotion_data, emotion_labels = load_data('/Users/kowshikbala/PycharmProjects/Facial Recognition and Emotion Detection Using Deep Learning/data/emotion_dataset')

    face_data, face_labels, emotion_data, emotion_labels = filter_datasets(face_data, face_labels, emotion_data, emotion_labels)

    # Step 1: Encode face labels (names of people) as integers
    face_label_encoder = LabelEncoder()
    face_labels_encoded = face_label_encoder.fit_transform(face_labels)
    face_labels_one_hot = to_categorical(face_labels_encoded)

    # Step 2: Encode emotion labels as integers
    emotion_label_encoder = LabelEncoder()
    emotion_labels_encoded = emotion_label_encoder.fit_transform(emotion_labels)
    emotion_labels_one_hot = to_categorical(emotion_labels_encoded)

    # Save the label encoders
    with open('checkpoints/face_label_encoder.pkl', 'wb') as f:
        pickle.dump(face_label_encoder, f)
    with open('checkpoints/emotion_label_encoder.pkl', 'wb') as f:
        pickle.dump(emotion_label_encoder, f)

    # Get the number of unique classes for people and emotions
    num_people = len(np.unique(face_labels_encoded))
    num_emotions = len(np.unique(emotion_labels_encoded))

    # Build and compile the model
    model = build_model(num_people, num_emotions)
    model.compile(optimizer='adam',
                  loss={'face_output': 'categorical_crossentropy', 'emotion_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    print(f"Face Data Shape: {face_data.shape}, Face Labels Shape: {face_labels.shape}")
    print(f"Emotion Data Shape: {emotion_data.shape}, Emotion Labels Shape: {emotion_labels.shape}")

    # Train the model
    model.fit(
        {'input_layer': face_data},  # Use the correct input layer name
        {'face_output': face_labels_one_hot, 'emotion_output': emotion_labels_one_hot},
        epochs=20,
        batch_size=32
    )

    # Save the trained model
    model.save('checkpoints/facial_emotion_recognition_model.h5')
    return model


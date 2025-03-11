from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context
#request.install_opener(request.build_opener(request.HTTPSHandler(context=ssl_context)))
def build_model(num_people, num_emotions):
    # Load base VGG16 model pre-trained on ImageNet
    base_model = VGG16( input_shape=(128, 128, 3), include_top=False, weights='imagenet',)
    x = Flatten()( base_model.output )

    # Branch for face recognition
    face_output = Dense(num_people, activation='softmax', name='face_output')(x)

    # Branch for emotion detection
    emotion_output = Dense(num_emotions, activation='softmax', name='emotion_output')(x)

    # Create final model
    model = Model( inputs=base_model.input, outputs=[face_output, emotion_output] )
    return model

   # model.compile(optimizer='adam',loss={'face_output': 'categorical_crossentropy', 'emotion_output': 'categorical_crossentropy'},metrics=['accuracy'])


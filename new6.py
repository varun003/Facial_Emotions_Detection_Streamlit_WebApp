### Webcam feed using streamlit app

import cv2
import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Facial Emotion Analysis")
# run = st.checkbox('Run')
run = st.button('Start')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained model outside the loop
new_model = tf.keras.models.load_model('final_model.h5')

while run:
     
    _, frame = camera.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    faces = faceCascade.detectMultiScale(frame, 1.1, 4)
    

    for x, y, w, h in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h),(255,255,255), 2)
        face_roi = frame[y:y+h, x:x+w]
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image,axis=-1) # add channel dimension
        final_image = np.expand_dims(final_image, axis=0) # add batch dimension
        final_image = final_image / 255.0
            
        Predictions = new_model.predict(final_image)
        emotion_label = np.argmax(Predictions)

        if emotion_label == 0:
            status = 'Angry'
            color = (0, 0, 255)
        elif emotion_label == 1:
            status = 'Disgust'
            color = (0, 0, 255)
        elif emotion_label == 2:
            status = 'Fear'
            color = (0, 0, 255)
        elif emotion_label == 3:
            status = 'Happy'
            color = (0, 255, 0)
        elif emotion_label == 4:
            status = 'Neutral'
            color = (0, 255, 255)
        elif emotion_label == 5:
            status = 'Sad'
            color = (255, 0, 0)
        else:
            status = 'Surprise'
            color = (255, 255, 0)
                
        cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        
    FRAME_WINDOW.image(frame)

else:
    st.write('Stopped')
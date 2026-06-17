import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("emotion_detector.keras")

emotion_map = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Sad',
    5:'Surprise',
    6:'Neutral'
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

st.title("Facial Emotion Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.error("No face detected")

    else:

        x,y,w,h = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(48,48))

        face = face / 255.0

        face = face.reshape(1,48,48,1)

        pred = model.predict(face,verbose=0)

        emotion = emotion_map[np.argmax(pred)]

        confidence = np.max(pred)*100

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        st.image(img)

        st.success(
            f"Emotion: {emotion}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        st.bar_chart(pred[0])
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_detector.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

emotion_map = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Sad',
    5:'Surprise',
    6:'Neutral'
}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48,48))

        face = face / 255.0

        face = face.reshape(1,48,48,1)

        pred = model.predict(face, verbose=0)

        idx = np.argmax(pred)
        emotion = emotion_map[idx]

        confidence = np.max(pred) * 100

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

        cv2.putText(
            frame,
            f"{emotion} {confidence:.1f}%",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )
        print(pred[0])
        print("Predicted:", emotion_map[np.argmax(pred)])

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
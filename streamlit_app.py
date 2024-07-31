


import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json
import os
import time
import pandas as pd

class StreamlitApp:
    def __init__(self, root_path):
        self.root_path = root_path
        self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = self.load_model()
        self.face_cascade = cv2.CascadeClassifier(os.path.join(self.root_path, 'haarcascade_frontalface_default.xml'))
        self.emotion_counts = {emotion: 0 for emotion in self.text_list}

    def load_model(self):
        model_json_file = os.path.join(self.root_path, 'model.json')
        model_weights_file = os.path.join(self.root_path, 'model_weights.h5')  

        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_weights_file)

        return loaded_model

    def predict_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        predictions = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            roi = cv2.resize(face, (56, 56))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            pred = self.model.predict(roi)
            text_idx = np.argmax(pred)
            text = self.text_list[text_idx]
            predictions.append((x, y, w, h, text))
            self.emotion_counts[text] += 1

        return predictions

    def calculate_emotion_percentages(self):
        total_predictions = sum(self.emotion_counts.values())
        if total_predictions == 0:
            return {emotion: 0 for emotion in self.text_list}
        return {emotion: (count / total_predictions) * 100 for emotion, count in self.emotion_counts.items()}

    def run(self):
        st.title("Mental Health Monitoring System")

        run = st.checkbox('Start')

        col1, col2 = st.columns(2)

        with col1:
            st.header("Real-Time Face Monitoring")
            FRAME_WINDOW = st.image([])
            emotion_text = st.empty()

        with col2:
            st.header("Emotion Distribution")
            chart_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break

            predictions = self.predict_emotion(frame)

            for (x, y, w, h, text) in predictions:
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if predictions:
                _, _, _, _, text = predictions[0]
                emotion_text.markdown(f"**Emotion: {text}**")
            else:
                emotion_text.markdown(f"**Emotion: Processing..**")


            emotion_percentages = self.calculate_emotion_percentages()
            emotion_data = pd.DataFrame(emotion_percentages.items(), columns=['Emotion', 'Percentage'])
            chart_placeholder.bar_chart(emotion_data.set_index('Emotion'),horizontal=True,color="#FF0000")

            time.sleep(0.1)  

        cap.release()

if __name__ == "__main__":
    root_path = os.path.dirname(__file__)
    app = StreamlitApp(root_path)
    app.run()


#////////////////////////////////////////////////////////////////////////////////////////////////



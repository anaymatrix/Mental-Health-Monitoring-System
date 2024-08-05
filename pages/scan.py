
import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json
import os
import time
# import datetime
import pandas as pd
from datetime import datetime
import altair as alt

# st.session_state['start'] = False


root_path = os.path.dirname(__file__)
image_path = os.path.join(root_path,'AI-LAB-LOGO.png')
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.page_link("home.py",label="Home")
st.sidebar.page_link("pages/sign_up.py",label="Sign Up")
st.sidebar.page_link("pages/log_in.py",label="Login")
st.sidebar.page_link("pages/scan.py",label="Scan")
st.sidebar.page_link("pages/history.py",label="History")



class StreamlitApp:
    def __init__(self, root_path):
        self.root_path = root_path
        self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.model = self.load_model()
        self.face_cascade = cv2.CascadeClassifier(os.path.join(self.root_path, 'haarcascade_frontalface_default.xml'))
        self.emotion_counts = {emotion: 0 for emotion in self.text_list}
        self.emotion_history = pd.DataFrame(columns=['Timestamp', 'Emotion'])

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

    def save_emotion_history(self, file):
        # Save history to a CSV file
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{st.session_state['username']}_{timestamp}.csv"
        history_file = os.path.join(self.root_path, 'history/',filename)
        file.to_csv(history_file, index=False)
        # saved_df = pd.DataFrame(file)
        # st.write(f"Emotion history saved to {history_file}.")
        # st.write(saved_df)
        st.write("All records are saved..")


    def run(self):

        if st.session_state['login']:

            st.title("Mental Health Monitoring System")
            st.session_state['start'] = False

            # run = st.button('Start')
            # stop = st.button('Stop')

            col1, col2 = st.columns(2)

            with col1:
                run = st.button('Start')

                st.header("Real-Time Face Monitoring")
                FRAME_WINDOW = st.image([])
                emotion_text = st.empty()
                stop = st.button('Stop')
                st.text(" ")
                st.text(" ")

                if st.button("Show History"):
                    st.switch_page("pages/history.py")

            with col2:
                st.text(" ")
                st.text(" ")
                st.text(" ")
                st.text(" ")

                st.header("Emotion Analysis Over Time")
                chart_placeholder = st.empty()

            cap = cv2.VideoCapture(0)

            while run and not stop :
                ret, frame = cap.read()
                if not ret:
                    break

                predictions = self.predict_emotion(frame)

                for (x, y, w, h, text) in predictions:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    new_row = pd.DataFrame({'Timestamp': [timestamp], 'Emotion': [text]})
                    self.emotion_history = pd.concat([self.emotion_history, new_row], ignore_index=True)

                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                st.session_state["emotion_history"] = self.emotion_history
                st.session_state['start'] = True
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if predictions:
                    _, _, _, _, text = predictions[0]
                    emotion_text.markdown(f"**Emotion: {text}**")
                else:
                    emotion_text.markdown(f"**Emotion: Processing..**")

                if not self.emotion_history.empty:
                    emotion_counts = self.emotion_history['Emotion'].value_counts().reset_index()
                    emotion_counts.columns = ['Emotion', 'Count']
                    
                    chart = alt.Chart(emotion_counts).mark_bar().encode(
                        x='Emotion',
                        y='Count',
                        color='Emotion',
                        tooltip=['Emotion', 'Count']
                    ).properties(width=400, height=300)

                    chart_placeholder.altair_chart(chart, use_container_width=True)

                # time.sleep(0.1)

            cap.release()

            if stop and st.session_state['start'] :
                self.save_emotion_history(st.session_state["emotion_history"])

        else:
            st.warning('Please Login first.', icon="⚠️")
            if st.button("Login"):
                st.switch_page("pages/log_in.py")

if __name__ == "__main__":
    if "login" not in st.session_state:
        st.switch_page(os.getcwd()+"/home.py")
    root_path = os.path.dirname(__file__)
    app = StreamlitApp(root_path)
    app.run()




#//////////////////////////////////////////////////////////////////////////////////////////////////////

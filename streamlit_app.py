


# import streamlit as st
# import numpy as np
# import cv2
# from keras.models import model_from_json
# import os
# import time
# import pandas as pd

# class StreamlitApp:
#     def __init__(self, root_path):
#         self.root_path = root_path
#         self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#         self.model = self.load_model()
#         self.face_cascade = cv2.CascadeClassifier(os.path.join(self.root_path, 'haarcascade_frontalface_default.xml'))
#         self.emotion_counts = {emotion: 0 for emotion in self.text_list}

#     def load_model(self):
#         model_json_file = os.path.join(self.root_path, 'model.json')
#         model_weights_file = os.path.join(self.root_path, 'model_weights.h5')  

#         with open(model_json_file, "r") as json_file:
#             loaded_model_json = json_file.read()
#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights(model_weights_file)

#         return loaded_model

#     def predict_emotion(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         predictions = []
#         for (x, y, w, h) in faces:
#             face = gray[y:y+h, x:x+w]
#             roi = cv2.resize(face, (56, 56))
#             roi = roi.astype('float32') / 255.0
#             roi = np.expand_dims(roi, axis=0)
#             roi = np.expand_dims(roi, axis=-1)

#             pred = self.model.predict(roi)
#             text_idx = np.argmax(pred)
#             text = self.text_list[text_idx]
#             predictions.append((x, y, w, h, text))
#             self.emotion_counts[text] += 1

#         return predictions

#     def calculate_emotion_percentages(self):
#         total_predictions = sum(self.emotion_counts.values())
#         if total_predictions == 0:
#             return {emotion: 0 for emotion in self.text_list}
#         return {emotion: (count / total_predictions) * 100 for emotion, count in self.emotion_counts.items()}

#     def run(self):
#         st.title("Mental Health Monitoring System")

#         run = st.checkbox('Start')

#         col1, col2 = st.columns(2)

#         with col1:
#             st.header("Real-Time Face Monitoring")
#             FRAME_WINDOW = st.image([])
#             emotion_text = st.empty()

#         with col2:
#             st.header("Emotion Distribution")
#             chart_placeholder = st.empty()

#         cap = cv2.VideoCapture(0)
#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             predictions = self.predict_emotion(frame)

#             for (x, y, w, h, text) in predictions:
#                 cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             if predictions:
#                 _, _, _, _, text = predictions[0]
#                 emotion_text.markdown(f"**Emotion: {text}**")
#             else:
#                 emotion_text.markdown(f"**Emotion: Processing..**")


#             emotion_percentages = self.calculate_emotion_percentages()
#             emotion_data = pd.DataFrame(emotion_percentages.items(), columns=['Emotion', 'Percentage'])
#             chart_placeholder.bar_chart(emotion_data.set_index('Emotion'),horizontal=True,color="#FF0000")

#             time.sleep(0.1)  

#         cap.release()

# if __name__ == "__main__":
#     root_path = os.path.dirname(__file__)
#     app = StreamlitApp(root_path)
#     app.run()


# #////////////////////////////////////////////////////////////////////////////////////////////////



# import streamlit as st
# import numpy as np
# import cv2
# from keras.models import model_from_json
# import os
# import time
# import pandas as pd
# from datetime import datetime

# class StreamlitApp:
#     def __init__(self, root_path):
#         self.root_path = root_path
#         self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#         self.model = self.load_model()
#         self.face_cascade = cv2.CascadeClassifier(os.path.join(self.root_path, 'haarcascade_frontalface_default.xml'))
#         self.emotion_counts = {emotion: 0 for emotion in self.text_list}
#         self.emotion_history = pd.DataFrame(columns=['Timestamp', 'Emotion'])

#     def load_model(self):
#         model_json_file = os.path.join(self.root_path, 'model.json')
#         model_weights_file = os.path.join(self.root_path, 'model_weights.h5')  

#         with open(model_json_file, "r") as json_file:
#             loaded_model_json = json_file.read()
#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights(model_weights_file)

#         return loaded_model

#     def predict_emotion(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         predictions = []
#         for (x, y, w, h) in faces:
#             face = gray[y:y+h, x:x+w]
#             roi = cv2.resize(face, (56, 56))
#             roi = roi.astype('float32') / 255.0
#             roi = np.expand_dims(roi, axis=0)
#             roi = np.expand_dims(roi, axis=-1)

#             pred = self.model.predict(roi)
#             text_idx = np.argmax(pred)
#             text = self.text_list[text_idx]
#             predictions.append((x, y, w, h, text))
#             self.emotion_counts[text] += 1

#         return predictions

#     def calculate_emotion_percentages(self):
#         total_predictions = sum(self.emotion_counts.values())
#         if total_predictions == 0:
#             return {emotion: 0 for emotion in self.text_list}
#         return {emotion: (count / total_predictions) * 100 for emotion, count in self.emotion_counts.items()}

#     def save_emotion_history(self):
#         # Save history to a CSV file
#         history_file = os.path.join(self.root_path, 'emotion_history.csv')
#         self.emotion_history.to_csv(history_file, index=False)
#         st.write(f"Emotion history saved to {history_file}.")
#         st.write("Data saved:", self.emotion_history)

#     def run(self):
#         st.title("Mental Health Monitoring System")

#         # Add Start and Stop buttons
#         run = st.checkbox('Start')
#         stop = st.button('Stop')

#         col1, col2 = st.columns(2)

#         with col1:
#             st.header("Real-Time Face Monitoring")
#             FRAME_WINDOW = st.image([])
#             emotion_text = st.empty()

#         with col2:
#             st.header("Emotion Analysis Over Time")
#             chart_placeholder = st.empty()

#         cap = cv2.VideoCapture(0)
#         while run and not stop:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             predictions = self.predict_emotion(frame)

#             for (x, y, w, h, text) in predictions:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
#                 # Use pd.concat to add a new row to the DataFrame
#                 new_row = pd.DataFrame({'Timestamp': [timestamp], 'Emotion': [text]})
#                 self.emotion_history = pd.concat([self.emotion_history, new_row], ignore_index=True)

#                 cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             if predictions:
#                 _, _, _, _, text = predictions[0]
#                 emotion_text.markdown(f"**Emotion: {text}**")
#             else:
#                 emotion_text.markdown(f"**Emotion: Processing..**")

#             # Update the chart with the latest emotion history
#             # Group by the timestamp and emotion to count occurrences
#             emotion_counts = self.emotion_history.groupby(['Timestamp', 'Emotion']).size().unstack(fill_value=0)
#             chart_placeholder.line_chart(emotion_counts, use_container_width=True)

#             time.sleep(0.1)

#         cap.release()

#         # Save the emotion history when "Stop" is clicked
#         if stop:
#             self.save_emotion_history()

# if __name__ == "__main__":
#     root_path = os.path.dirname(__file__)
#     app = StreamlitApp(root_path)
#     app.run()



#///////////////////////////////////////////////////////////////////////////////////////////////////////////






# import streamlit as st
# import numpy as np
# import cv2
# from keras.models import model_from_json
# import os
# import time
# import pandas as pd
# from datetime import datetime

# class StreamlitApp:
#     def __init__(self, root_path):
#         self.root_path = root_path
#         self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#         self.model = self.load_model()
#         self.face_cascade = cv2.CascadeClassifier(os.path.join(self.root_path, 'haarcascade_frontalface_default.xml'))
#         self.emotion_counts = {emotion: 0 for emotion in self.text_list}
#         self.emotion_history = pd.DataFrame(columns=['Timestamp', 'Emotion'])

#     def load_model(self):
#         model_json_file = os.path.join(self.root_path, 'model.json')
#         model_weights_file = os.path.join(self.root_path, 'model_weights.h5')  

#         with open(model_json_file, "r") as json_file:
#             loaded_model_json = json_file.read()
#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights(model_weights_file)

#         return loaded_model

#     def predict_emotion(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         predictions = []
#         for (x, y, w, h) in faces:
#             face = gray[y:y+h, x:x+w]
#             roi = cv2.resize(face, (56, 56))
#             roi = roi.astype('float32') / 255.0
#             roi = np.expand_dims(roi, axis=0)
#             roi = np.expand_dims(roi, axis=-1)

#             pred = self.model.predict(roi)
#             text_idx = np.argmax(pred)
#             text = self.text_list[text_idx]
#             predictions.append((x, y, w, h, text))
#             self.emotion_counts[text] += 1

#         return predictions

#     def calculate_emotion_percentages(self):
#         total_predictions = sum(self.emotion_counts.values())
#         if total_predictions == 0:
#             return {emotion: 0 for emotion in self.text_list}
#         return {emotion: (count / total_predictions) * 100 for emotion, count in self.emotion_counts.items()}

#     def save_emotion_history(self,file):
#         # Save history to a CSV file
#         history_file = os.path.join(self.root_path, 'emotion_history.csv')
#         # st.session_state["emotion_history"].to_csv(history_file, index=False)
#         file.to_csv(history_file, index=False)
#         saved_df = pd.DataFrame(file)
#         st.write(f"Emotion history saved to {file}.")
#         st.write(saved_df)


#     def run(self):
#         st.title("Mental Health Monitoring System")

#         run = st.button('Start')
#         stop = st.button('Stop')

#         col1, col2 = st.columns(2)

#         with col1:
#             st.header("Real-Time Face Monitoring")
#             FRAME_WINDOW = st.image([])
#             emotion_text = st.empty()

#         with col2:
#             st.header("Emotion Analysis Over Time")
#             chart_placeholder = st.empty()

#         cap = cv2.VideoCapture(0)


#         while run and not stop:
            
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             predictions = self.predict_emotion(frame)
            
#             for (x, y, w, h, text) in predictions:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
#                 new_row = pd.DataFrame({'Timestamp': [timestamp], 'Emotion': [text]})
#                 self.emotion_history = pd.concat([self.emotion_history, new_row], ignore_index=True)
                
#                 # print(new_row['Timestamp'])
#                 print(self.emotion_history.shape,"11111111111111111111111111111111111")
#                 # print(self.emotion_history['Emotion'])

#                 cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             st.session_state["emotion_history"] = self.emotion_history
#             # print("AFter for ",st.session_state["emotion_history"])
#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


#             if predictions:
#                 _, _, _, _, text = predictions[0]
#                 emotion_text.markdown(f"**Emotion: {text}**")
#             else:
#                 emotion_text.markdown(f"**Emotion: Processing..**")

#             # Group by the timestamp and emotion to count occurrences
#             emotion_counts = self.emotion_history.groupby(['Timestamp', 'Emotion']).size().unstack(fill_value=0)
#             chart_placeholder.line_chart(emotion_counts, use_container_width=True)

#             time.sleep(0.1)
        
#         cap.release()

#         # print(self.emotion_history.shape,"333333333333333333333333333333333333")
#         if stop:
#             # print(self.emotion_history.shape,"44444444444444444444444444444444444444444")
#             self.save_emotion_history(st.session_state["emotion_history"])
#             # print(self.emotion_history.shape,"5555555555555555555555555555555555555555")
        

# if __name__ == "__main__":
#     root_path = os.path.dirname(__file__)
#     app = StreamlitApp(root_path)
#     app.run()




#///////////////////////////////////////////////////////////////////////////////////////////////////////////





import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json
import os
import time
import pandas as pd
from datetime import datetime
import altair as alt


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
        history_file = os.path.join(self.root_path, 'emotion_history.csv')
        file.to_csv(history_file, index=False)
        saved_df = pd.DataFrame(file)
        # st.write(f"Emotion history saved to {history_file}.")
        st.write(saved_df)

    def run(self):

        st.title("Mental Health Monitoring System")

        run = st.button('Start')
        stop = st.button('Stop')

        col1, col2 = st.columns(2)

        with col1:
            st.header("Real-Time Face Monitoring")
            FRAME_WINDOW = st.image([])
            emotion_text = st.empty()

        with col2:
            st.header("Emotion Analysis Over Time")
            chart_placeholder = st.empty()

        cap = cv2.VideoCapture(0)

        while run and not stop:
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
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if predictions:
                _, _, _, _, text = predictions[0]
                emotion_text.markdown(f"**Emotion: {text}**")
            else:
                emotion_text.markdown(f"**Emotion: Processing..**")

            # Create a bar chart with Altair
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

            time.sleep(0.1)

        cap.release()

        if stop:
            self.save_emotion_history(st.session_state["emotion_history"])

if __name__ == "__main__":
    root_path = os.path.dirname(__file__)
    app = StreamlitApp(root_path)
    app.run()




#//////////////////////////////////////////////////////////////////////////////////////////////////////




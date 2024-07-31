# import os
# import numpy as np
# import cv2
# from keras.models import model_from_json
# from collections import deque


# class EmotionPredictor:
#     def __init__(self, root_path):
#         self.root_path = root_path
#         self.model = self.load_model()
#         self.face_cascade = cv2.CascadeClassifier(os.path.join(root_path, 'haarcascade_frontalface_default.xml'))
#         self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#     def load_model(self):
#         model_json_file = os.path.join(self.root_path, 'model.json')
#         model_weights_file = os.path.join(self.root_path, 'model_weights.h5')

#         with open(model_json_file, "r") as json_file:
#             loaded_model_json = json_file.read()
#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights(model_weights_file)

#         return loaded_model

#     def predict_emotion(self):
#         cap = cv2.VideoCapture(0)
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#             for (x, y, w, h) in faces:
#                 face = gray[y:y+h, x:x+w]
#                 roi = cv2.resize(face, (56, 56))
#                 roi = roi.astype('float32') / 255.0
#                 roi = np.expand_dims(roi, axis=0)
#                 roi = np.expand_dims(roi, axis=-1)

#                 pred = self.model.predict(roi)
#                 text_idx = np.argmax(pred)
#                 text = self.text_list[text_idx]

#                 cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             cv2.imshow("Emotion Recognition", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()



# if __name__ == "__main__" : 

#     root_path = os.path.dirname(__file__)
#     EmotionPredictor(root_path=root_path).predict_emotion()

#///////////////////////////////////////////////////////////////////////////////////////////



import os
import numpy as np
import cv2
from keras.models import model_from_json


class EmotionPredictor:
    def __init__(self, root_path):
        self.root_path = root_path
        self.model = self.load_model()
        self.face_cascade = cv2.CascadeClassifier(os.path.join(root_path, 'haarcascade_frontalface_default.xml'))
        self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def load_model(self):
        model_json_file = os.path.join(self.root_path, 'model.json')
        model_weights_file = os.path.join(self.root_path, 'model_weights.h5')

        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_weights_file)

        return loaded_model

    def predict_emotion(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                roi = cv2.resize(face, (56, 56))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                # Debug: Print the ROI shape and values
                # print(f"ROI shape: {roi.shape}, ROI values: {roi}")

                pred = self.model.predict(roi)
                text_idx = np.argmax(pred)
                text = self.text_list[text_idx]

                # Debug: Print prediction values and the selected emotion
                print(f"Predictions: {pred}, Selected Emotion: {text}")

                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root_path = os.path.dirname(__file__)
    EmotionPredictor(root_path=root_path).predict_emotion()




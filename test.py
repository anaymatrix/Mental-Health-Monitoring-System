from predict import EmotionPredictor
import os

root_path = os.path.dirname(__file__)
EmotionPredictor(root_path=root_path).predict_emotion()


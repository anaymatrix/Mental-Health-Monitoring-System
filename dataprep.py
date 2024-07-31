import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img

class DataPreparation:
    def __init__(self, base_path, pic_size=48, batch_size=128):
        self.base_path = base_path
        self.pic_size = pic_size
        self.batch_size = batch_size

    def prepare_data(self):
        plt.figure(0, figsize=(12, 20))
        cpt = 0
        for expression in os.listdir(self.base_path + "train"):
            for i in range(1, 6):
                cpt += 1
                plt.subplot(7, 5, cpt)
                img = load_img(self.base_path + "train/" + expression + "/" + os.listdir(self.base_path + "train/" + expression)[i], target_size=(self.pic_size, self.pic_size))
                plt.imshow(img, cmap="gray")
        plt.tight_layout()

    def create_generators(self):
        train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, rotation_range=20, horizontal_flip=True)
        validation_datagen = ImageDataGenerator(rescale=1.0/255)

        train_generator = train_datagen.flow_from_directory(self.base_path + "train", target_size=(56, 56), color_mode="grayscale", batch_size=self.batch_size, class_mode='categorical', shuffle=True)
        validation_generator = validation_datagen.flow_from_directory(self.base_path + "validation", target_size=(56, 56), color_mode="grayscale", batch_size=self.batch_size, class_mode='categorical', shuffle=False)

        return train_generator, validation_generator



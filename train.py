from keras.callbacks import ModelCheckpoint
from dataprep import DataPreparation
from model import CNNModel
from keras.models import model_from_json


class ModelTrainer:
    def __init__(self, base_path, epochs=60, batch_size=128):
        self.base_path = base_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataprep = DataPreparation(base_path, batch_size=batch_size)
        self.model = CNNModel().get_model()

    def train(self):
        train_generator, validation_generator = self.dataprep.create_generators()

        checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=train_generator.n // train_generator.batch_size,
                                           epochs=self.epochs,
                                           validation_data=validation_generator,
                                           validation_steps=validation_generator.n // validation_generator.batch_size,
                                           callbacks=callbacks_list
                                        #    use_multiprocessing=True
                                           )
        
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)


        return history
    

    
    # def save_model(self):
    #     model_json = self.model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)






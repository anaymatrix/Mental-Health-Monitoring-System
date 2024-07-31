from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Nadam


class CNNModel:
    def __init__(self, input_shape=(56, 56, 1), nb_classes=7, lr=0.0001):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.lr = lr
        self.model = self.build_model()

    def build_model(self):
        
        # model = Sequential()
        # model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(128, (5, 5), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(512, (3, 3), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Conv2D(512, (3, 3), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())

        # model.add(Dense(256))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        # model.add(Dense(512))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        # model.add(Dense(self.nb_classes, activation='softmax'))

        # opt = Adam(lr=self.lr)
        # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # return model


        #---------------------------------------------------------------------------------------

        model = Sequential()

        # conv 1
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=self.input_shape))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        # conv 2
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        # conv 3
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        # conv 4
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        # conv 5
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        # conv 6
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('elu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())

        # dence 1
        model.add(Dense(128))
        model.add(Activation('elu'))

        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # dence 2
        model.add(Dense(self.nb_classes, activation='softmax'))

        opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model



    def get_model(self):
        return self.model



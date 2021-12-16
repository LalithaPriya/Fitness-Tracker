from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sensordatawindow import SensorDataWindow
from keras.optimizers import Adam
from scipy import signal
import numpy as np
from SensorDataPoints import SensorDataPoints
import cv2

class ExerciseClassifier:

    def __init__(self, weights_file='exercise_classification_weights.h5'):
        loadOnly = True
        updateFromPrevBatch = False
        img_width, img_height = 100, 10
        input_shape = (img_width, img_height, 6)
        chanDim = -1
        self.minWindowLength = 5
        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(8, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3))
        self.model.add(Activation('sigmoid'))

        self.model.summary()

        print("Loading from: "+ weights_file)
        self.model.load_weights(weights_file)

    def getspectrogram(self, x, fs):
        # print(len(x))
        fs = int(fs)
        f, t, Sxx = signal.spectrogram(x, fs, nfft = 1000, return_onesided=True, nperseg =fs//2, noverlap = 0 )
        # print(Sxx.shape)
        Sxx = Sxx[0:100]
        return cv2.resize(Sxx, dsize=(10,100), interpolation=cv2.INTER_CUBIC)

    def getSpectrogramsFromAllChannels(self, datapoints, frequency, nchannels):
        print("freq: "+str(frequency))
        return [self.getspectrogram(datapoints[:,channel], frequency) for channel in range(nchannels)]

    def getSpectrogramsFromAllSensors(self, allsensorDatapoints, sensorNChannels):
        allSensorSpectrograms = []
        for sensorDataPoints, nchannels in zip(allsensorDatapoints, sensorNChannels):
            fs = sensorDataPoints.frequency
            allSensorSpectrograms.extend(self.getSpectrogramsFromAllChannels(sensorDataPoints.datapoints, sensorDataPoints.frequency, nchannels))
            print("sensor spectrogram shape: "+str(allSensorSpectrograms[-1].shape))
        return np.dstack(allSensorSpectrograms)

    def classifySensorData(self, sensorDataWindows, sensorNChannels):
        allsensorDatapoints = []
        for sensorDataWindow in sensorDataWindows:
            allsensorDatapoints.append(sensorDataWindow.getWindow())
        # print(allsensorDatapoints[0].frequency, len(allsensorDatapoints[0].datapoints), self.minWindowLength*.95)
        if len(allsensorDatapoints[0].datapoints)/allsensorDatapoints[0].frequency < self.minWindowLength*.95:
            return -10
        spectrogramInput = self.getSpectrogramsFromAllSensors(allsensorDatapoints, sensorNChannels)
        print("spectrogram input shape: "+str(spectrogramInput.shape))
        return np.argmax(self.model.predict(spectrogramInput[np.newaxis,:,:,:]))





# for randomIndex in random.sample(range(0, spectrorams.shape[0]), nb_validation_samples):
#     y_predict = 
#     print(randomIndex, y_predict, labels_catogerical[randomIndex], np.argmax(y_predict)==labels[randomIndex])
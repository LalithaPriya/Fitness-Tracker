from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sensordatawindow import SensorDataWindow
import numpy as np
from SensorDataPoints import SensorDataPoints
import scipy.interpolate as interp

class SeriesExerciseClassifier:

    def __init__(self, weights_file='exercise_classification_series_weights.h5', windowLength=3000):
        loadOnly = True
        updateFromPrevBatch = False
        img_width, img_height = 600, 1
        input_shape = (img_width, img_height, 6)
        print(input_shape)
        chanDim = -1
        self.windowLength = windowLength
        self.model = Sequential()
        self.model.add(Conv2D(16, (6, 1), input_shape=input_shape))
        self.model.add(Activation('relu'))
        # self.model.add(BatchNormalizat on(axis=chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))

        self.model.add(Conv2D(8, (4, 1)))
        self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization(axis=chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))

        self.model.add(Conv2D(4, (4, 1)))
        self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization(axis=chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))

        # self.model.add(Conv2D(64, (3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization(axis=chanDim))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3))
        self.model.add(Activation('sigmoid'))

        self.model.summary()

        print("Loading from: "+ weights_file)
        self.model.load_weights(weights_file)


    def getNearestMultipleOf(self, num, multiplier): # (399,200)=400 (542,20) = 540
        div = num/(1.0* multiplier)
        return int(multiplier*np.round(div))


    def getseries(self, x, maxDatapointsForSensor):
        # print(len(x))
        # f, t, Sxx = signal.series(x, fs, nfft = 1000, retursn_onesided=True, nperseg =fs//2, noverlap = 0 )
        # print(Sxx.shape)
        #print(len(x))
        xinterp = interp.interp1d(np.arange(x.shape[0]),x)
        x = xinterp(np.linspace(0,x.shape[0]-1,maxDatapointsForSensor))
        print(x[:, np.newaxis].shape)

        return x[:,np.newaxis]

    def getSeriesFromAllChannels(self, datapoints, nchannels, maxDatapointsForSensor):
        return [self.getseries(datapoints[:,channel], maxDatapointsForSensor) for channel in range(nchannels)]

    def getSeriesFromAllSensors(self, allsensorDatapoints,sensorNChannels, preview = False):
        window = (0, self.windowLength*1.0/1000)
        allSensorSeries = []
        maxDatapointsForSensor = np.max([self.getNearestMultipleOf(int(window[1]*fs)-int(window[0]*fs), self.getNearestMultipleOf(fs*(window[1]-window[0]),100) ) for fs in [sensorDataPoints.frequency for sensorDataPoints in allsensorDatapoints]])
    #print(maxDatapointsForSensor)
        for sensorDataPoints, nchannels in zip(allsensorDatapoints, sensorNChannels):
            fs = sensorDataPoints.frequency
            diff = self.getNearestMultipleOf(int(window[1]*fs)-int(window[0]*fs), self.getNearestMultipleOf(fs*(window[1]-window[0]),100) )
            #print(int(window[0]*fs),int(window[0]*fs)+diff)
            allSensorSeries.extend(self.getSeriesFromAllChannels(sensorDataPoints.datapoints[int(window[0]*fs):int(window[0]*fs)+diff], nchannels, maxDatapointsForSensor))
        if preview :
            for series,i in zip(allSensorSeries, range(len(allSensorSeries))):
                plt.figure(i)
                plt.imshow(series)
            plt.show()
        return np.dstack(allSensorSeries)


    def classifySensorData(self, sensorDataWindows, sensorNChannels):
        allsensorDatapoints = []
        for sensorDataWindow in sensorDataWindows:
            allsensorDatapoints.append(sensorDataWindow.getWindow())
        print(len(allsensorDatapoints[1].datapoints))
        if len(allsensorDatapoints[0].datapoints)/allsensorDatapoints[0].frequency < self.windowLength*.95*.001:
            return -10
        seriesInput = self.getSeriesFromAllSensors(allsensorDatapoints, sensorNChannels)
        print("series input shape: "+str(seriesInput.shape))
        return np.argmax(self.model.predict(seriesInput[np.newaxis,:,:,:]))





# for randomIndex in random.sample(range(0, spectrorams.shape[0]), nb_validation_samples):
#   y_predict = 
#   print(randomIndex, y_predict, labels_catogerical[randomIndex], np.argmax(y_predict)==labels[randomIndex])
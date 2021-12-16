import sys
from flask import Flask
from flask_sockets import Sockets
import json
from sensordatawindow import SensorDataWindow 
from exercise_classifier import ExerciseClassifier
from series_exercise_classifier import SeriesExerciseClassifier
import matplotlib.pyplot as plt
import cv2
import time
import os

clear = lambda : os.system('clear')

useSeriesClassifier =False

app = Flask(__name__)
sockets = Sockets(app)

if(useSeriesClassifier):
	classifier = SeriesExerciseClassifier( weights_file='exercise_classification_series_weights.h5')
	window_size = 3000
else:
	classifier = ExerciseClassifier( weights_file='exercise_classification_weights.h5')
	window_size = 5000

loading = ['|','/','--','\\','|','/','--','\\']

def getExercise(number):
	exercisemap = {
		0: 'Arms horizontal Rotation',
		1: 'Arms Up down movement',
		2: 'Jogging'
	}
	return '\n'+'\n'.join([str(key+1)+")" + exercisemap[key] + ("<---- this is being done" if key==number else "") for key in exercisemap.keys()]) if number in exercisemap.keys() else "\nCollecting Buffer for first few seconds\n"
def getImageFile(number):
	exercisemap = {
		0: '/home/kalyan/Desktop/myfitnessclassifier/armswing.png',
		1: '/home/kalyan/Desktop/myfitnessclassifier/armupdown.jpg',
		2: '/home/kalyan/Desktop/myfitnessclassifier/Jogging.jpg'
	}
	return exercisemap.get(number, "/home/kalyan/Desktop/myfitnessclassifier/nothng.png")

@sockets.route('/putData')
def echo_socket(ws):
	lasensorwindow = SensorDataWindow(window_size)
	gysensorwindow = SensorDataWindow(window_size)
	i=0
	while True:
		message = ws.receive()
		resp = ""
		la = None
		gy = None
		try:
			sensorData = json.loads(message)
			# print(sensorData['sensorListMap'])
			resp = "received batch: "+str(sensorData['batch'])
			la = sensorData['sensorListMap']['Linear_Acceleration']
			gy = sensorData['sensorListMap']['Gyroscope_Calibrated']
			# print("Current window size", sensorwindow.putBatch(sensorData['sensorListMap']['Linear_Accelertion']))
		except ValueError, err:  # includes simplejson.decoder.JSONDecodeError
			resp = message
			# print(err)
			la = None
			gy = None
			print(""+message)
		ws.send(message)
		if la is not None and gy is not None:
			# print(la)
			i = i+1
			clear()
			print("Current la window size", lasensorwindow.putBatch(la) )
			print("Current gy window size", gysensorwindow.putBatch(gy) )
			exercise = classifier.classifySensorData([gysensorwindow, lasensorwindow],[3,3])
			print("\n\n\n Exercise: " +loading[i%8]+ getExercise(exercise))
			print("\n")
	print("socket closed")




@app.route('/')
def hello():
	return 'Hello World!'

if __name__ == "__main__":
	from gevent import pywsgi
	from geventwebsocket.handler import WebSocketHandler
	server = pywsgi.WSGIServer(('192.168.0.200', 5000), app, handler_class=WebSocketHandler)
	server.serve_forever()

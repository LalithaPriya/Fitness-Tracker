import Queue
from SensorDataPoints import SensorDataPoints
import numpy as np

class SensorDataWindow:

	def __init__(self, period):
		self.period = period*1000000 # ms to ns
		self.queue = Queue.Queue()

	def getItemTime(self, item):
		return int(float(item[0]))


	def putBatch(self, sensorDataBatch):
		for sensorData in sensorDataBatch:
			self.queue.put(sensorData)
		latest = self.getItemTime(sensorDataBatch[len(sensorDataBatch)-1])
		
		item = self.queue.get()
		while latest - self.getItemTime(item) > self.period:
			item = self.queue.get()
		return self.queue.qsize()

	def getWindow(self):
		datapointsWithTime=list(self.queue.queue)
		datapoints = []
		for datapoint in datapointsWithTime:
			datapoints.append(datapoint[1:])
		return SensorDataPoints(np.asarray(datapoints), len(datapoints)/((datapointsWithTime[-1][0]-datapointsWithTime[0][0])/1000000000))

	def getWindowLength(self):
		return self.period/1000000





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionNet(nn.Module):
	def __init__(self):
		super(PredictionNet, self).__init__()
		# input is 3 numbers in a tensor: [price, time, predictNDays]
		# price is the current price at the time
		# predictNDays is the number of days from the current time to predict
		# output the predicted price
		self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=1, batch_first=True),
		self.lin = nn.Linear(64, 1)
	def forward(self, x):
		x = self.lstm(x)
		x = self.lin(x)
		return x

def datetime_to_timestamp(d):
	# convert datetime to unix timestamp
	import time, datetime
	# YYYY-MM-DD HH:MM:SS
	return int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))
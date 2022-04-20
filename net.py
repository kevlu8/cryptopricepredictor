import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import requests
import os

# update data
# first delete all files in data/
# os.system("rm data/*")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionNet(nn.Module):
	def __init__(self):
		super(PredictionNet, self).__init__()
		# input is 1 tensor: [allPrices]
		# allPrices is a tensor of all prices
		# output is a tensor of predicted prices for the next 1024 days 
		self.fc1 = nn.Linear(1, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1)
		self.dropout = nn.Dropout(0.2)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = x.view(-1, 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x

# Q: What does "mat1 and mat2" shapes cannot be multiplied (1x32 and 1x32) mean?
# A: The shapes cannot be multiplied because the first dimension of mat1 and mat2 is 1.
# Q: So how do we fix it?
# A: We can use "view" function to change the shape of tensor.
# Q: What does "view" function do?
# A: It changes the shape of tensor.
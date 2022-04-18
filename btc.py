import net
import os

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--train", help="Train the model", action="store_true")
argparser.add_argument("--epochs", help="Amount of epochs", type=int, default=10000)
argparser.add_argument("--no_cuda", help="Don't use cuda", action="store_true")
args = argparser.parse_args()

if not args.no_cuda and net.torch.cuda.is_available():
	net.device = net.torch.device("cuda")
else:
	net.device = net.torch.device("cpu")

btcpredict = net.PredictionNet().to(net.device)
optim = net.optim.Adam(btcpredict.parameters(), lr=0.01)

if os.path.exists("models/btc.pt"):
	c = net.torch.load("models/btc.pt")
	btcpredict.load_state_dict(c["model"])
	optim.load_state_dict(c["optim"])
	print("Loaded model")

class Loader(net.torch.utils.data.Dataset):
	# we can ignore number,name,symbol,high,low,open,volume,market_cap because we don't need it
	# we only need date and close
	# ignore the first row because it's the header
	def __init__(self, filename):
		self.data = []
		with open(str(filename), "r") as f:
			for line in f:
				if line.startswith("SNo"):
					continue # first line is header
				date = line.split(",")[3]
				close = float(line.split(",")[7])
				self.data.append((date, close))
		# self.data.reverse()
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]

if args.train:
	if os.path.exists("data/btc.csv"): # contains all bitcoin data in such a format: number,name,symbol,date,high,low,open,close,volume,market_cap
		print("Loading data...")
		# load data from csv file using Loader class
		loader = Loader("data/btc.csv")
		# create dataloader for training and testing data
		train_loader = net.torch.utils.data.DataLoader(loader, batch_size=32, shuffle=True)
		# train the model
		print("Training model...")
		# input is 3 numbers in a tensor: [price, time, predictNDays]
        # price is the current price at the time
        # predictNDays is the number of days from the current time to predict
        # output the predicted price
		for epoch in range(args.epochs):
			# measure time for each epoch
			start = net.time.time()
			for i, (date, close) in enumerate(train_loader):
				optim.zero_grad()
				# convert datetime to unix timestamp
				# YYYY-MM-DD HH:MM:SS
				timestamp = net.datetime_to_timestamp(date[i])
				# convert close to tensor
				close_b = net.torch.tensor(close[i]).to(net.device)
				# convert timestamp to tensor
				timestamp = net.torch.tensor(timestamp).to(net.device)
				# predict 1 day
				predictNDays = 1
				# create input tensor
				input = net.torch.tensor([[close_b, timestamp, predictNDays]]).to(net.device)
				# forward pass
				prediction = btcpredict(input.float())
				# calculate loss
				loss = net.nn.MSELoss()(prediction, close_b)
				# backward pass
				loss.backward()
				optim.step()
			# time
			end = net.time.time()
			# print progress
			if epoch % 100 == 0:
				print("Epoch: {}, Loss: {}, Time: {}s".format(epoch, loss.item(), round(end-start, 3)))
				net.torch.save({"model": btcpredict.state_dict(), "optim": optim.state_dict()}, "models/btc.pt")
	else:
		print("No data found")
else:
	if os.path.exists("data/btc.csv"):
		print("Loading data...")
		# load data from csv file using Loader class
		loader = Loader("data/btc.csv")
		# create dataloader for training and testing data
		test_loader = net.torch.utils.data.DataLoader(loader, batch_size=32, shuffle=True)
		# test model
		print("Testing model...")
		for i, (date, close) in enumerate(test_loader):
			# convert date to tensor
			date = net.torch.tensor(date, dtype=net.torch.float)
			# convert close to tensor
			close = net.torch.tensor(close, dtype=net.torch.float)
			# reshape close to (batch_size, 1)
			close = close.reshape(close.shape[0], 1)
			# forward pass
			pred = btcpredict(date)
			# calculate loss
			loss = net.torch.nn.MSELoss()(pred, close)
			if i % 100 == 0:
				print("Prediction: {}".format(pred.item()))
				print("Close: {}".format(close.item()))
				print("")
	else:
		print("No data found")
from numpy import dtype
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
				close = float(line.split(",")[7])
				self.data.append(close)
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
		train_loader = net.torch.utils.data.DataLoader(loader, batch_size=512)
		# train the model
		print("Training model...")
		# input is 1 tensor: [allPrices]
		# allPrices is a tensor of all prices
		# output is a tensor of predicted prices for the next 512 days
		for epoch in range(args.epochs):
			# measure time for each epoch
			start = net.time.time()
			for i, close in enumerate(train_loader):
				if len(close) != 512:
					continue
				optim.zero_grad()
				# split close into 2 tensors of equal size
				close_a = close[:256]
				close_b = close[256:]
				# convert close to tensor
				close_a = net.torch.tensor(close_a, device=net.device, dtype=net.torch.float32)
				close_b = net.torch.tensor(close_b, device=net.device, dtype=net.torch.float32)
				# start training
				prediction = btcpredict(close_a)
				# print(prediction)
				# calculate loss
				# convert prediction from [512, 1] to [512]
				prediction = prediction.view(-1)
				close_b = close_b.view(-1)
				# print(prediction)
				loss = net.F.mse_loss(prediction, close_b)
				# backpropagate
				loss.backward()
				optim.step()
			# time
			end = net.time.time()
			# print progress
			if epoch % 100 == 0:
				try:
					print("Epoch: {}, Loss: {}, Time/Epoch: {}s".format(epoch, loss.item(), round(end-start, 3)))
				except NameError:
					print("Fuck")
				net.torch.save({"model": btcpredict.state_dict(), "optim": optim.state_dict()}, "models/btc.pt")
		net.torch.save({"model": btcpredict.state_dict(), "optim": optim.state_dict()}, "models/btc.pt")
	else:
		print("No data found")
else:
	if os.path.exists("data/btc.csv"):
		print("Loading data...")
		# load data from csv file using Loader class
		loader = Loader("data/btc.csv")
		# get last 512 prices
		last_512 = loader.data[-512:]
		# convert to tensor
		last_512 = net.torch.tensor(last_512, device=net.device, dtype=net.torch.float32)
		# test model
		print("Using model to predict next prices...")
		# predict
		btcpredict.eval()
		prediction = btcpredict(last_512)
		# print prediction
		predictions = prediction.detach().cpu().numpy().tolist()
		print("Predictions: {}".format(predictions))
		import matplotlib.pyplot as plt
		plt.plot(predictions)
		plt.title("BTC Price over next 512 days")
		plt.show()
	else:
		print("No data found")
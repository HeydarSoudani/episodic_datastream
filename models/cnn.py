import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
	def __init__(self, args):
		super(CNNEncoder, self).__init__()
		self.args = args
		self.device = None

		img_channels = 3

		self.layer1 = nn.Sequential(
		nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
		nn.Dropout(args.dropout)
		)
		self.layer2 = nn.Sequential(
		nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  #input : 14 * 14 * 32, output: 12 * 12 * 64
		nn.BatchNorm2d(64),
		nn.ReLU(),
		nn.MaxPool2d(2),  #12 * 12 * 64, output: 6 * 6 * 64
		nn.Dropout(args.dropout)
		)
		self.fc1 = nn.Linear(in_features=64*6*6, out_features=args.hidden_dims)
		
		self.classifier = nn.Linear(args.hidden_dims, 10)
		
		# self.device = device
		# self.to(device)

	def forward(self, samples):
		out = self.layer1(samples)            
		cnn_output = self.layer2(out)
		out = cnn_output.view(cnn_output.size(0), -1)
		features = torch.relu(self.fc1(out))
		logits = self.classifier(features)
		return logits, features
	
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		self.layer1 = self.layer1.to(*args, **kwargs)
		self.layer2 = self.layer2.to(*args, **kwargs)
		self.fc1 = self.fc1.to(*args, **kwargs)
		self.classifier = self.classifier.to(*args, **kwargs)
		# self.weight = self.weight.to(*args, **kwargs)
		return self
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)


# Layers 1&2, with kernel_size 5
# Layers 3&4, with kernel_size 3 
class Conv_4(nn.Module):
	def __init__(self, args):
		super(Conv_4, self).__init__()
		
		if args.dataset in ['mnist', 'rmnist', 'fmnist', 'pfmnist', 'rfmnist']:
			img_channels = 1	  	# 1
			self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
		elif args.dataset in ['cifar10', 'cifar100']:
			img_channels = 3	  	# 3 
			self.last_layer = 2 	# 4 for 3-layers - 2 for 4-layers

		self.filters_length = 256    # 128 for 3-layers - 256 for 4-layers

		self.layer1 = nn.Sequential(
			nn.Conv2d(img_channels, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
			# nn.ReLU(),
			nn.PReLU(),
			nn.Conv2d(32, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
			nn.BatchNorm2d(32),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
			nn.Dropout(args.dropout)
		)
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, padding=2), #input: 14 * 14 * 32, output: 14 * 14 * 64
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=5, padding=2), #input: 14 * 14 * 64, output: 14 * 14 * 64
			nn.BatchNorm2d(64),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 14 * 14 * 64, output: 7* 7 * 64
			nn.Dropout(args.dropout)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1), #input: 7 * 7 * 64, output: 7 * 7 * 128
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1), #input: 7 * 7 * 128, output: 7 * 7 * 128
			nn.BatchNorm2d(128),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 7 * 7 * 128, output: 3* 3 * 128
			nn.Dropout(args.dropout)
		)
		
		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1), #input: 3 * 3 * 128, output: 3 * 3 * 256
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), #input: 3*3*256, output: 3*3*256
			nn.BatchNorm2d(256),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 3*3*256, output: 1*1*256
			nn.Dropout(args.dropout)
		)

		self.ip1 = nn.Linear(self.filters_length*self.last_layer*self.last_layer, args.hidden_dims)
		self.preluip1 = nn.PReLU()
		self.dropoutip1 = nn.Dropout(args.dropout)
		self.classifier = nn.Linear(args.hidden_dims, 10)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)

		features = self.preluip1(self.ip1(x))
		x = self.dropoutip1(features)
		logits = self.classifier(x)
		
		return logits, features

	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		
		self.layer1 = self.layer1.to(*args, **kwargs)
		self.layer2 = self.layer2.to(*args, **kwargs)
		self.layer3 = self.layer3.to(*args, **kwargs)
		self.layer4 = self.layer4.to(*args, **kwargs)

		self.ip1 = self.ip1.to(*args, **kwargs)
		self.preluip1 = self.preluip1.to(*args, **kwargs)
		self.dropoutip1 = self.dropoutip1.to(*args, **kwargs)
		self.classifier = self.classifier.to(*args, **kwargs)
		return self

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)
  



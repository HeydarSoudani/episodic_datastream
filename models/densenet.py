import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        return torch.cat([x, out], dim=1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()

        inter_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.dropout(self.conv2(self.relu(self.bn2(out))))
        return torch.cat([x, out], dim=1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)
        self.pooling = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        return self.pooling(self.dropout(self.conv1(self.relu(self.bn1(x)))))

class DenseBlock(nn.Module):
    def __init__(self, number_layers, in_channels, block, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()

        layers = []

        for i in range(number_layers):
            layers.append(block(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, drop_rate=drop_rate))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
	def __init__(self, args, tensor_view, number_layers=6, growth_rate=12, reduction=2, bottleneck=True, drop_rate=0.0):
		super(DenseNet, self).__init__()

		assert len(tensor_view) == 3

		channels = 2 * growth_rate

		if bottleneck:
				block = BottleneckBlock
		else:
				block = BasicBlock

		# 1st conv before any dense block
		self.conv1 = nn.Conv2d(in_channels=tensor_view[0], out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)

		# 1st block
		self.block1 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
		channels = channels + number_layers * growth_rate
		self.trans1 = TransitionBlock(channels, channels // reduction, drop_rate)
		channels = channels // reduction

		# 2nd block
		self.block2 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
		channels = channels + number_layers * growth_rate
		self.trans2 = TransitionBlock(channels, channels // reduction, drop_rate)
		channels = channels // reduction

		# 3rd block
		self.block3 = DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
		channels = channels + number_layers * growth_rate

		# global average pooling and classifier
		self.bn = nn.BatchNorm2d(channels)
		self.relu = nn.ReLU(inplace=True)
		self.pooling = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

		self.fc1 = nn.Linear(channels * ceil(tensor_view[1] / 8) * ceil(tensor_view[2] / 8), 10000)
		self.fc2 = nn.Linear(10000, args.hidden_dims)

		self.cls_dropout = nn.Dropout(args.dropout)
		# self.weight = nn.Parameter(torch.randn(args.seen_labels, args.hidden_dims)) #[5, 768]
		self.fc3 = nn.Linear(args.hidden_dims, 10)

		self.channels = channels
		self.tensor_view = tensor_view

		# self.device = device
		# self.to(device)

	@property
	def shape(self):
			return torch.Size((self.channels, ceil(self.tensor_view[1] / 8), ceil(self.tensor_view[2] / 8)))

	def forward(self, samples):
		
		out = self.conv1(samples)
		out = self.trans1(self.block1(out))
		out = self.trans2(self.block2(out))
		out = self.block3(out)
		out = self.relu(self.bn(out))
		out = self.pooling(out).view(out.size(0), -1)
		out = self.relu(self.fc1(out))
		feature = self.fc2(out)
		feature_relu = self.relu(feature)
		pooled_output = self.cls_dropout(feature_relu)
		logits = self.fc3(pooled_output)
		# logits = F.linear(pooled_output, self.weight)

		return logits, feature_relu

	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		
		self.conv1 = self.conv1.to(*args, **kwargs)
		self.trans1 = self.trans1.to(*args, **kwargs)
		self.trans2 = self.trans2.to(*args, **kwargs)
		self.block3 = self.block3.to(*args, **kwargs)
		self.bn = self.bn.to(*args, **kwargs)
		self.pooling = self.pooling.to(*args, **kwargs)
		self.fc1 = self.fc1.to(*args, **kwargs)
		self.fc2 = self.fc2.to(*args, **kwargs)
		self.fc3 = self.fc3.to(*args, **kwargs)

		self.cls_dropout = self.cls_dropout.to(*args, **kwargs)
		# self.weight = self.weight.to(*args, **kwargs)
		return self

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)

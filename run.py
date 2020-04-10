from dataloader import DatasetLoader
from network import Net
from test import evaluate
from train import train

dataset = DatasetLoader()
dataset.transform_load()

dataset.display_samples(dataset.TRAIN)
dataset.display_samples(dataset.VAL)
dataset.display_samples(dataset.TEST)

# create network, modify, and set parameters
net = Net(dataset)
net.build()
net.set_params()

evaluate(net, dataset)

net, history = train(net, dataset)

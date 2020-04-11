import os
from dataloader import DatasetLoader
from network import Net
from test import evaluate
from train import train
from config import cfg
from utils import plot_loss, plot_accuracy
import torch


dataset = DatasetLoader()
dataset.transform_load()

# create network, modify, and set parameters
net = Net(dataset)
net.build()
net.set_params()

print("net =", net)

#evaluate(net, dataset)

if not os.path.exists(cfg.MODEL.DIR):
    os.makedirs(cfg.MODEL.DIR)

if cfg.MODEL.CONTINUE_TRAINING and os.path.exists(cfg.MODEL.FILENAME):
    net.model = torch.load(cfg.MODEL.FILENAME)

net, history = train(net, dataset)

print("net =", net)

# save history
history.to_pickle(cfg.MODEL.HISTORY_PATH)

# save model
torch.save(net.model, cfg.MODEL.FILENAME)

plot_accuracy(history)
plot_loss(history)

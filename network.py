from config import cfg
from collections.abc import Iterable
from torchvision import models
import torch.nn as nn
from torchsummary import summary
from dataloader import DatasetLoader
import torch.optim as optim


class Net:
    def __init__(self, dataset):
        self.model = None
        self.freeze_layer_till = 28
        self.dataset = dataset

        # Set model parameters
        self.criterion = None
        self.optimizer = None

    def build(self):
        self.download_vgg()
        self.freeze_parameters()
        self.summary()
        self.modify_vgg()
        self.summary()

    def set_params(self):
        self.criterion = cfg.MODEL.CRITERION
        if cfg.MODEL.OPTIMIZER == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.MODEL.LR)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.MODEL.LR, momentum=0.5)

    def download_vgg(self):
        self.model = models.vgg16(pretrained=True)
        self.model = self.model.to(cfg.CONST.DEVICE)

    def freeze_parameters(self):
        ct = 0
        for child in self.model.children():
            if isinstance(child, Iterable):
                for layer in child:
                    ct += 1
                    if ct < self.freeze_layer_till:
                        for param in layer.parameters():
                            param.requires_grad = False
            else:
                ct += 1

    def summary(self):
        summary(self.model, (3, cfg.TRANSFORM.RESIZE, cfg.TRANSFORM.RESIZE))

    def modify_vgg(self):

        num_features = self.model.classifier[0].in_features
        # remove classification layers
        features = list(self.model.classifier.children())[:-7]

        features.extend([nn.Linear(num_features, 1000)])
        features.extend([nn.ReLU(inplace=True)])
        features.extend([nn.Dropout(p=0.5, inplace=False)])
        features.extend([nn.Linear(1000, 1000)])  # Add our layer with 2 outputs
        features.extend([nn.ReLU(inplace=True)])
        features.extend([nn.Dropout(p=0.5, inplace=False)])
        features.extend([nn.Linear(1000, len(self.dataset.label_names))])
        features.extend([nn.LogSoftmax(dim=1)])
        self.model.classifier = nn.Sequential(*features)

        self.model = self.model.to(cfg.CONST.DEVICE)
        print("VGG Architecture Modified!")


def main():
    dataset = DatasetLoader()
    dataset.transform_load()

    net = Net(dataset)
    net.build()
    net.set_params()

if __name__ == "__main__":
    main()
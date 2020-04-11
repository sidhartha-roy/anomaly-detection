import os
import numpy as np
from config import cfg
from utils import plot_images
import split_folders
import torch
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader


class DatasetLoader:
    def __init__(self):
        self.data_transforms = None
        self.image_datasets = None
        self.dataloaders = None
        self.dataset_sizes = None
        self.label_names = None
        self.TRAIN = cfg.CONST.TRAIN
        self.VAL = cfg.CONST.VAL
        self.TEST = cfg.CONST.TEST
        self.weighted_sampler = None

    def transform_load(self):
        self.transform()
        self.build_image_datasets()
        self.build_dataloaders()
        if cfg.DATASETS.WEIGHTED_SAMPLER:
            self.weighted_loader()

    def transform(self):
        self.data_transforms = {
            self.TRAIN: transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=cfg.TRANSFORM.BRIGHTNESS,
                                       contrast=cfg.TRANSFORM.CONTRAST,
                                       saturation=cfg.TRANSFORM.SATURATION,
                                       hue=cfg.TRANSFORM.HUE),
                transforms.RandomRotation(cfg.TRANSFORM.DEGREES, expand=True),
                transforms.RandomAffine(cfg.TRANSFORM.DEGREES, translate=None, scale=None,
                                        shear=None, resample=False, fillcolor=0),
                transforms.Resize(cfg.TRANSFORM.RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.TRANSFORM.MEAN, std=cfg.TRANSFORM.STD)
            ]),
            self.VAL: transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(cfg.TRANSFORM.RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.TRANSFORM.MEAN, std=cfg.TRANSFORM.STD)
            ]),
            self.TEST: transforms.Compose([
                transforms.Resize(cfg.TRANSFORM.RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.TRANSFORM.MEAN, std=cfg.TRANSFORM.STD)
            ])
        }

    def build_image_datasets(self):
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(cfg.DATASETS.PATH, x),
                transform=self.data_transforms[x]
            )
            for x in [self.TRAIN, self.VAL, self.TEST]
        }

    def build_dataloaders(self):
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x], batch_size=cfg.DATASETS.BATCH_SIZE,
                shuffle=True, num_workers=cfg.CONST.NUM_WORKERS
            )
            for x in [self.TRAIN, self.VAL, self.TEST]
        }

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in [self.TRAIN, self.VAL, self.TEST]}

        for x in [self.TRAIN, self.VAL, self.TEST]:
            print("Loaded {} images under {}".format(self.dataset_sizes[x], x))

        print("Classes: ")
        self.label_names = self.image_datasets[self.TRAIN].classes
        print(self.image_datasets[self.TRAIN].classes)

    def weighted_loader(self):
        targets = np.array(self.dataloaders[self.TRAIN].dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        for i in range(len(classes)):
            print("Classes = {}, counts {}".format(classes[i], class_counts[i]))
        num_classes = len(classes)

        class_weights = 1. / class_counts
        samples_weight = np.array([class_weights[t] for t in targets])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        self.weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        self.dataloaders[self.TRAIN] = DataLoader(self.image_datasets[self.TRAIN],
                                                  batch_size=cfg.DATASETS.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  sampler=self.weighted_sampler)

    def split_folder(self):
        """
        Split with a ratio.
        To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
        :return:
        """
        split_folders.ratio(cfg.DATASETS.RAW_PATH,
                            output=cfg.DATASETS.PATH,
                            seed=cfg.CONST.RANDOM_SEED,
                            ratio=(cfg.DATASETS.TRAIN_RATIO, cfg.DATASETS.VAL_RATIO, cfg.DATASETS.TEST_RATIO))  # default values

    def display_samples(self, dataset_type):
        sample_loader = torch.utils.data.DataLoader(
            self.image_datasets[dataset_type], batch_size=9, shuffle=True,
            num_workers=cfg.CONST.NUM_WORKERS, pin_memory=cfg.CONST.PIN_MEMORY,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels, self.label_names)


def main():
    dataset = DatasetLoader()
    dataset.transform_load()

if __name__ == "__main__":
    main()
from torch.utils.data import DataLoader, random_split

from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100


class Datasets:
    def __init__(self, dataset_name, dataset_config):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        dataset = eval(dataset_name)(dataset_config)
        self.train_dataset, self.test_dataset = dataset.get_dataset()

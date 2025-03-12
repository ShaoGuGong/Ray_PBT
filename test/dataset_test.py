import os
import sys
import torchvision


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

data_dir = "~/Documents/dataset/"

data_dir = os.path.expanduser(data_dir)

torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)


# from trial import Hyperparameter
# from utils import ModelType, get_data_loader
#
# hyper = Hyperparameter(0.1, 0.1, 64, ModelType.RESNET_18)
# get_data_loader(hyper.model_type, hyper.batch_size)

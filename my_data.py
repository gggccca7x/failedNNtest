import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# ------------------------------------ MNIST START -------------------------------------------------

def getMNIST_Trainloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)),
    ])

    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Label	Description
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
def getClothingTrainloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)),
    ])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# ------------------------------------ MNIST END -------------------------------------------------

class LetterDataset(Dataset):
    def __init__(self, dataNPArray, transform=None, transformInputs=None):
        # data loading
        self.dataNPArray = dataNPArray
        self.X = dataNPArray[: , 1:]
        self.y = dataNPArray[: , [0]]
        self.n_samples = dataNPArray.shape[0]
        self.transform = transform
        self.transformInputs = transformInputs

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples 

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs).float(), torch.from_numpy(targets).long()

class Normalize:
    def __call__(self, sample):
        inputs, target = sample
        inputs /= 255
        return inputs, target

# inputs csv file and outputs numpy array representation of that file
def getLines(filename):
    csvFile = pd.read_csv(filename, delimiter=',') # add/remove nrows=1000 parameter to test quickly
    csvFileNP = np.array(csvFile)
    np.random.shuffle(csvFileNP)
    return csvFileNP[:300000, :], csvFileNP[300000:, :]

def getLettersTrainloader():
    npDataTrain, npDataValidation = getLines('reading_letters\\handwritten_data_785.csv')
    transform = torchvision.transforms.Compose([ToTensor(), Normalize()])
    datasetTrain = LetterDataset(npDataTrain, transform=transform)
    datasetValidation = LetterDataset(npDataValidation, transform=transform)
    dataloaderTrain = DataLoader(dataset=datasetTrain, batch_size=64, shuffle=True)
    dataloaderValidation = DataLoader(dataset=datasetValidation, batch_size=64, shuffle=True)
    return dataloaderTrain, dataloaderValidation
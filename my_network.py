import torch
from torch import nn
from torch import optim
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*3*3, 150)
        self.fc2 = nn.Linear(150, 26)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        print(x.shape)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64*3*3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.softmax(self.fc2(x))) # note: skipped out on softmax...?
        return x


def trainModel(epochs, model, trainloader, device, trainloader_validation):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)

    model.to(device)

    validation_loss_min = np.Inf

    for e in range(epochs):
        train_loss = 0
        validation_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            labels = labels.view(-1) # NOTE: had to add for letters dataset, mnist dataset had correct shape already
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for images, labels in trainloader_validation:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            labels = labels.view(-1)
            loss = criterion(output, labels)
            validation_loss += loss.item()

        print("Epoch: " , e)

        print(f"Training Loss: {train_loss/len(trainloader)}")
        print(f"Validation Loss: {validation_loss/len(trainloader_validation)}")

        if validation_loss <= validation_loss_min:
            print("Validation loss has decreased, saving model")
            torch.save(model.state_dict(), 'LettersReadingModel.pth')
            validation_loss_min = validation_loss

        print("---------------------------")

    return model
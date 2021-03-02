import torch
from torch import nn
from torchvision import datasets, transforms
from torch.nn import functional as F
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import my_network
import my_data
import os.path
from os import path

deviceCuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deviceCPU = torch.device('cpu')
model = my_network.NN()

if path.exists('LettersReadingModel.pth'):
    print("load existing model")
    model.load_state_dict(torch.load('LettersReadingModel.pth'))
else:
    print("no existing model so create and train new one")
    trainloader, trainloader_validation = my_data.getLettersTrainloader()
    dataiter = iter(trainloader)
    epochs = 50
    model = my_network.trainModel(epochs, model, trainloader, deviceCuda, trainloader_validation)


def convertImage(image):
    image = image[:,:,:3]
    image = 1- np.mean(image, axis=2)
    paddingAmount = max(28-image.shape[0],28-image.shape[1])
    image = np.pad(image, (0,paddingAmount))
    return image[:28, :28]

model.to(deviceCPU)
torch.no_grad()
model.eval()

trainloader, trainloader_validation = my_data.getLettersTrainloader()
dataiter = iter(trainloader)
images, _ = dataiter.next()

# imgInvoice = mpimg.imread('invoice2.png')
# img_cropped_invoice = imgInvoice[515:538, 355:370, :]
# imageInvoice = convertImage(img_cropped_invoice)
# imageInvoice = torch.Tensor(imageInvoice)
# imageInvoice = imageInvoice.view(1, -1)

plt.imshow(images[0].view(28,28,1,-1).numpy().squeeze(), cmap='Greys_r')
# plt.imshow(imageInvoice.view(28,28,1,-1).numpy().squeeze(), cmap='Greys_r')

# print("test image")
# print(images[0])
# print(images[0].shape)
result = model.forward(images)
ps = torch.exp(result)

# print("test A")
# print(imageInvoice[0])
# print(imageInvoice.shape)
# resultInvoice = model.forward(imageInvoice)
# ps = torch.exp(resultInvoice)

print(ps[0])
top_p, top_class = ps.topk(1, dim=1)
print(top_class[:1, :])
plt.show()

# trainloader = my_data.getLettersTrainloader()
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.shape)
# result = model.forward(images)
# prediction = F.softmax(result, dim=1)
# final_pred = np.argmax(prediction.detach().numpy(), axis=1)
# print(final_pred[0])

'''
# Visual test
trainloader = my_data.getLettersTrainloader()
dataiter = iter(trainloader)
torch.no_grad()
model.eval()
images, labels = dataiter.next()
image = images[0].view(28, 28, 1, -1)
plt.imshow(image.numpy().squeeze(), cmap='Greys_r')
result = model.forward(images)
ps = torch.exp(result)
top_p, top_class = ps.topk(1, dim=1)
# print(top_class[:10, :])
equals = top_class == labels.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')

prediction = F.softmax(result, dim=1)
final_pred = np.argmax(prediction.detach().numpy(), axis=1)
print(final_pred[0])
plt.show()
'''
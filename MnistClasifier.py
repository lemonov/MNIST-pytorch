import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from torch.autograd import Variable
MODEL_PATH = 'model_mnist.tar'

class Unbuffered(object):
    def __init__(self, stream):
       self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x 
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

imsize = 64
loader = transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])



transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

resize = transforms.Resize(28)


def image_loader(image_name):
    image = Image.open(image_name).convert('L')
    image = image.resize((28, 28))
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train():
    print("Training")

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False)
    print("Datasets loaded")

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    net = MnistClassifier()
    criterion = nn.CrossEntropyLoss()
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_list = list()
    for epoch in range(2):  # loop over the dataset multiple times
        print("Epoch ", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                loss_list.append(running_loss)

    plt.plot(loss_list)
    plt.show()


    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print("Saving model")
    torch.save(net, MODEL_PATH)
    print('Finished Training')

def recognize(image):
    image = image_loader(image)
    model = torch.load(MODEL_PATH)
    result = model(image)
    best = torch.argmax(result).item()
    confidence = F.softmax(result, dim = -1)

    return (best, confidence)


def main():
    # train()
    recognize("test.jpg")


if __name__ == '__main__':
    main()
    


# Network todolist
# Get samples or use loader (NCHW format)
# Create network structure
# Select loss function (CrossEntropyLoss/MSE/et.c)
# Select optimizer (StochasticGradientDescent)
# Iterate over epochs 
# validate output with targets
# calculate loss
# back-prop
#
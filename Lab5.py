import torch.nn as nn
import torch
import torch.optim as optim
import struct as st
import os
import random
import math
from PIL import Image
import random

def read_idx(archivo):
    data = open(archivo, 'rb')

    # magic_number = int.from_bytes(data.read(4), byteorder="big", signed=True)
    data.seek(0)
    magic = st.unpack('>4B', data.read(4))
    # print(magic[3])

    if magic[3] == 3:
        images = int.from_bytes(data.read(4), byteorder="big", signed=True)

        rows = int.from_bytes(data.read(4), byteorder="big", signed=True)

        columns = int.from_bytes(data.read(4), byteorder="big", signed=True)

        '''print('Magic number: {}\nNumber of images: {}\nRows: {}\nColumns: {}'.format(
            magic,
            images,
            rows,
            columns
        ))'''

        binary_vector = data.read(images * rows * columns)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        tensor = tensor.view(images, rows, columns)
        # print(tensor)
        return tensor

    elif magic[3] == 1:
        labels = int.from_bytes(data.read(4), byteorder="big", signed=True)

        '''print('Magic number: {}\nNumber of labels: {}'.format(
            magic,
            labels,
        ))'''

        binary_vector = data.read(labels)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        # print("LABELS", tensor.view(labels))

        return tensor


def save_images(images):
    una = Image.new('L', (28, 28))
    for i in range(0, 5):
        una.putdata(list(images[i].view(-1)))  # el -1 convierte a una dimension
        una.show()
        una.save(str(i) + '.jpg')


def filter_data(images, labels, singleLabel):
    x = (labels == singleLabel)
    y = x.nonzero()
    nums = images[y]
    image = Image.new('L', (28, 28))
    image.putdata(list(nums[random.randint(0, nums.size()[0])].view(-1)))
    # image.show()
    image.save(os.path.join('./filter_data/' + str(singleLabel) + '.jpg'))
    return nums


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden0 = nn.Linear(100, 128)
        self.hidden1 = nn.Linear(128, 256)
        self.hidden2 = nn.Linear(256, 512)

        # Output layer,
        self.output = nn.Linear(512, 784)

        # Define LeakyReLU activation and Sigmoid output
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden0(x)
        x = self.leakyRelu(x)
        x = self.hidden1(x)
        x = self.leakyRelu(x)
        x = self.hidden2(x)
        x = self.leakyRelu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden0 = nn.Linear(784, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 128)

        # Output layer,
        self.output = nn.Linear(128, 1)

        # Define LeakyReLU activation and Sigmoid output
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden0(x)
        x = self.leakyRelu(x)
        x = self.hidden1(x)
        x = self.leakyRelu(x)
        x = self.hidden2(x)
        x = self.leakyRelu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def loss_fn(y, oov):
    dif = ((oov-y)**2).mean()
    return dif


'''def training_opt(n, model, images, labels, optimizer):
    labels = labels.long()
    loss_fn = nn.CrossEntropyLoss()
    images = images.float().view(-1, 784)
    for i in range(0, n):
        t_p = model(images)
        loss = loss_fn(t_p, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d, Loss %f' % (i, float(loss)))

    torch.save(model.state_dict(), './save/nn')'''


def train_discriminator(model, real, fake):
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.BCELoss()
    real = real.float()
    fake = fake.float()
    real = (real - real.mean())/255
    fake = (fake - fake.mean()) / 255
    r = random.random()
    if r >= 0.5:
        n1 = random.randrange(real.size()[0])
        n2 = random.randrange(n1)
        miniBatch = real[n2:n1]
        miniBatch = miniBatch.view(-1, 784)
        output = model(miniBatch)
        y = torch.ones(output.shape)
    else:
        n1 = random.randrange(fake.size()[0])
        n2 = random.randrange(n1)
        miniBatch = fake[n2:n1]
        miniBatch = miniBatch.view(-1, 784)
        output = model(miniBatch)
        y = torch.zeros(output.shape)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_generator(generador, discriminador, fake):
    fake = fake.float().view(-1, 784)
    optimizer = optim.Adam(generador.parameters(), lr=0.0002)
    loss_fn = nn.BCELoss()
    optimizer.zero_grad()
    prediction = discriminador(fake)
    y = torch.ones(prediction.shape)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
    print(loss)
    return loss


if __name__ == '__main__':
    imagesTraining = read_idx('train-images.idx3-ubyte')
    labelsTraining = read_idx('train-labels.idx1-ubyte')
    generador = Generator()
    discriminator = Discriminator()
    epochs = 200
    #for n in range(0, epochs):


    fakes = torch.randn(imagesTraining.shape)
    train_discriminator(discriminator, imagesTraining, fakes)
    train_generator(generador, discriminator, fakes)

import torch.nn as nn
import torch
import torch.optim as optim
import struct as st
import os
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


def train_discriminator(discriminador, optimizer, real, fake):
    '''loss_fn = nn.BCELoss()
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
        output = discriminador(miniBatch)
        y = torch.ones(output.shape)
    else:
        n1 = random.randrange(fake.size()[0])
        n2 = random.randrange(n1)
        miniBatch = fake[n2:n1]
        miniBatch = miniBatch.view(-1, 784)
        output = discriminador(miniBatch)
        y = torch.zeros(output.shape)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("end of train_discriminator") '''

    real = real.float()
    fake = fake.float()
    loss_fn = nn.BCELoss()
    N = real.size()[0]
    optimizer.zero_grad()

    # real data
    target = torch.ones(N)
    target = target.unsqueeze(1)
    prediction_real = discriminador(real)
    error_real = loss_fn(prediction_real, target)
    error_real.backward()

    # fake data
    prediction_fake = discriminador(fake)
    M = fake.size()[0]
    target_fake = torch.zeros(M)
    target_fake = target_fake.unsqueeze(1)
    error_fake = loss_fn(prediction_fake, target_fake)
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminador, optimizer, fake):
    '''loss_fn = nn.BCELoss()
    optimizer.zero_grad()
    prediction = discriminador(fake)
    y = torch.ones(prediction.shape)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
    print(loss)
    return loss
    '''
    loss_fn = nn.BCELoss()
    N = fake.size(0)
    optimizer.zero_grad()
    prediction = discriminador(fake)
    target = torch.ones(N)
    target = target.unsqueeze(1)
    error = loss_fn(prediction, target)
    error.backward()
    optimizer.step()

    return error


if __name__ == '__main__':
    reales = read_idx('train-images.idx3-ubyte')
    labelsTraining = read_idx('train-labels.idx1-ubyte')
    fakes = torch.randn(reales.shape).detach()

    data_loader = torch.utils.data.DataLoader(reales, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)

    generador = Generator()
    discriminator = Discriminator()

    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002)
    optimizerG = optim.Adam(generador.parameters(), lr=0.0002)

    real = reales.float()
    fake = fakes.float()
    real = (real - real.mean()) / 255
    fake = (fake - fake.mean()) / 255

    epochs = 200
    for i in range(0, epochs):
        for n_batch, (real_batch) in enumerate(data_loader):
            # train discriminador
            real_data = real_batch.view(-1, 784)
            fake_data = generador(torch.randn(real_data.size(0), 100)).detach()
            error, predReal, predFake = train_discriminator(discriminator, optimizerD, real_data, fake_data)
            # print("real: ", real_data.size(), ", fake: ", fake_data.size())

            # train generator
            # fake_data = generador(torch.randn(real_data.size(0), 100)).detach()
            errorGen = train_generator(discriminator, optimizerG, fake_data)

            test_noise = torch.randn(16, 100)
            if (n_batch) % 100 == 0:
                test_images = generador(test_noise).data
                # print("1 test images: ", test_images.size())
                # test_images = test_images.view(test_images.size(0), 1, 28, 28)
                # print("2 test images: ", test_images.size())
                imagen = Image.new('L', (28, 28))
                imagen.putdata(list(test_images))
                imagen.save(os.path.join('./generadas/' + str(n_batch) + '.jpg'))
            # imagen.show()
            print("Error Dis: ", error.float(), ", error gen: ", errorGen.float())
            print("Epoch: ", i)

    # print(n_batch)
    '''n1 = random.randrange(real.size()[0])
    n2 = random.randrange(n1)
    miniBatchReales = real[n2:n1]
    miniBatchReales = miniBatchReales.view(-1, 784)
    n1 = random.randrange(fake.size()[0])
    n2 = random.randrange(n1)
    miniBatchFakes = fake[n2:n1]
    miniBatchFakes = miniBatchFakes.view(-1, 784)
    error, predReal, predFake = train_discriminator(discriminator, optimizerD, miniBatchReales, miniBatchFakes)

    g_error = train_generator(discriminator, optimizerG, miniBatchFakes)

    test_noise = torch.randn(16, 100)
    testImages = vectors_to_images(generador(test_noise)).data.cpu()
    imagen = Image.new('L', (28, 28))
    imagen.putdata(list(generador(test_noise)[0].view(-1)))
    # imagen.show()
    imagen.save(os.path.join('./generadas/' + str(i) + "_" + str(j) + '.jpg'))'''

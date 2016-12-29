#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# Network definition
class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024, 10),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
    parser.add_argument('--name', '-n', type=str, default="1.png", help='file name')
    args = parser.parse_args()

    model = L.Classifier(CNN())

    myNumber = Image.open(args.name).convert("L")
    myNumber = 1.0 - np.asarray(myNumber, dtype="float32") / 255
    myNumber = myNumber.reshape((1,1,28,28))

    chainer.serializers.load_npz('cnn.model', model)

    # Results
    x = chainer.Variable(myNumber)
    v = model.predictor(x)
    print("fileName:", args.name, "predict:", np.argmax(v.data))

    # print (myNumber)
    draw_digit(myNumber)


def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size),range(size))
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

    plt.show()

if __name__ == '__main__':
    main()
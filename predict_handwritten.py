#!/usr/bin/env python
from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
    parser.add_argument('--name', '-n', type=str, default="1.png", help='file name')
    args = parser.parse_args()

    myNumber = Image.open(args.name).convert("L")
    myNumber = 1.0 - np.asarray(myNumber, dtype="float32") / 255
    myNumber = myNumber.reshape((1, 784))

    model = L.Classifier(MLP(args.unit, 10))
    chainer.serializers.load_npz('linear.model', model)

    # Results
    x = chainer.Variable(myNumber)
    v = model.predictor(x)
    print("fileName:", args.name, "predict:", np.argmax(v.data))

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
